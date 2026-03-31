"""Support for Volcengine ASR speech-to-text service."""
import asyncio
import logging
import json
import uuid
import struct
import time
import aiohttp # Import aiohttp

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechToTextEntity,
)
from homeassistant.components.stt.models import (
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.aiohttp_client import async_get_clientsession # For getting HA's client session

from .const import (
    DOMAIN,
    CONF_APP_ID,
    CONF_ACCESS_TOKEN,
    CONF_RESOURCE_ID,
    CONF_SERVICE_URL,
    CONF_LANGUAGE,
    CONF_AUDIO_FORMAT,
    CONF_AUDIO_RATE,
    CONF_AUDIO_BITS,
    CONF_AUDIO_CHANNEL,
    CONF_ENABLE_ITN,
    CONF_ENABLE_PUNC,
    CONF_RESULT_TYPE,
    CONF_SHOW_UTTERANCES,
    CONF_PERFORMANCE_MODE,
    CONF_END_WINDOW_SIZE,
    CONF_FORCE_TO_SPEECH_TIME,
    CONF_LOG_TEXT_CHANGE_ONLY,
    CONF_ENABLE_PERF_LOG,
    CONF_LOG_LEVEL,
    DEFAULT_SERVICE_URL,
    DEFAULT_LANGUAGE,
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_AUDIO_RATE,
    DEFAULT_AUDIO_BITS,
    DEFAULT_AUDIO_CHANNEL,
    DEFAULT_ENABLE_ITN,
    DEFAULT_ENABLE_PUNC,
    DEFAULT_RESULT_TYPE,
    DEFAULT_SHOW_UTTERANCES,
    DEFAULT_PERFORMANCE_MODE,
    DEFAULT_END_WINDOW_SIZE,
    DEFAULT_FORCE_TO_SPEECH_TIME,
    DEFAULT_LOG_TEXT_CHANGE_ONLY,
    DEFAULT_ENABLE_PERF_LOG,
    DEFAULT_LOG_LEVEL,
    PERF_AUDIO_BATCH_SIZE,
    PERF_RESPONSE_TIMEOUT_SEND,
    PERF_RESPONSE_TIMEOUT_FINAL,
    LOG_TAG_AUDIO_SEND,
    LOG_TAG_RESPONSE_RECEIVE,
    LOG_TAG_TEXT_EXTRACT,
    LOG_TAG_WEBSOCKET,
    LOG_TAG_VAD,
)

_LOGGER = logging.getLogger(__name__)

async def async_get_engine(hass: HomeAssistant, config: ConfigType, discovery_info: DiscoveryInfoType | None = None):
    """Set up Volcengine ASR STT component."""
    return VolcengineASRProvider(hass, hass.data[DOMAIN])

async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType, # platform config
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    """Set up the Volcengine ASR STT platform."""
    _LOGGER.info("Setting up Volcengine ASR STT platform")
    component_config = hass.data.get(DOMAIN)
    if not component_config:
        _LOGGER.error("Volcengine ASR main component config not found in hass.data")
        return
    
    async_add_entities([VolcengineASRProvider(hass, component_config)])
    _LOGGER.info("Volcengine ASR STT platform setup complete")


class VolcengineASRProvider(SpeechToTextEntity):
    """Volcengine ASR speech-to-text provider."""

    def __init__(self, hass: HomeAssistant, config: ConfigType) -> None:
        """Initialize the provider."""
        self.hass = hass
        self._config = config
        self._attr_name = "Volcengine ASR"
        self.name = "Volcengine ASR" 
        self._connect_id = str(uuid.uuid4())
        
        # 初始化性能日志配置
        self._enable_perf_log = self._config.get(CONF_ENABLE_PERF_LOG, DEFAULT_ENABLE_PERF_LOG)
        self._log_level = self._config.get(CONF_LOG_LEVEL, DEFAULT_LOG_LEVEL)
        
        # 初始化性能指标计时器
        self._process_start_time = 0
        self._first_audio_send_time = 0
        self._last_audio_send_time = 0
        self._first_resp_time = 0
        self._last_resp_time = 0
        self._first_text_time = 0
        self._text_extraction_count = 0
        self._audio_chunks_sent = 0
        self._audio_bytes_sent = 0
        self._responses_received = 0
        
        # 设置日志级别对应的函数
        self._log_funcs = {
            "debug": _LOGGER.debug,
            "info": _LOGGER.info,
            "warning": _LOGGER.warning,
            "error": _LOGGER.error
        }
        
    def _perf_log(self, tag, message):
        """记录性能相关日志"""
        if self._enable_perf_log:
            log_func = self._log_funcs.get(self._log_level, _LOGGER.info)
            elapsed = time.time() - self._process_start_time
            log_func(f"[PERF][{tag}][+{elapsed:.3f}s] {message}")

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return [self._config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)]

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]
    
    def _preprocess_payload(self, payload_bytes: bytes) -> bytes:
        """Preprocesses the payload to remove any non-JSON prefix."""
        try:
            json_start_index = payload_bytes.find(b"{")
            if json_start_index == -1:
                _LOGGER.warning(f"ASR response payload does not contain JSON start character 	{{	. Payload (hex): {payload_bytes.hex()}")
                return b""
            if json_start_index > 0:
                _LOGGER.debug(f"ASR response payload has prefix. Original (hex): {payload_bytes.hex()}, Stripped (hex): {payload_bytes[json_start_index:].hex()}")
            return payload_bytes[json_start_index:]
        except Exception as e:
            _LOGGER.error(f"Error during payload preprocessing: {e}. Original payload (hex): {payload_bytes.hex()}")
            return payload_bytes

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: asyncio.StreamReader
    ) -> SpeechResult:
        # 重置计时器
        self._process_start_time = time.time()
        self._first_audio_send_time = 0
        self._last_audio_send_time = 0
        self._first_resp_time = 0
        self._last_resp_time = 0
        self._first_text_time = 0
        self._text_extraction_count = 0
        self._audio_chunks_sent = 0
        self._audio_bytes_sent = 0
        self._responses_received = 0
        
        self._perf_log("PROCESS", f"开始处理音频流. Metadata: {metadata}")
        _LOGGER.debug(f"Processing audio stream with metadata: {metadata}")
        if not self.check_metadata(metadata):
            _LOGGER.error(
                f"Unsupported audio metadata: format={metadata.format}, "
                f"codec={metadata.codec}, rate={metadata.sample_rate}, "
                f"bits={metadata.bit_rate}, channels={metadata.channel}. "
                f"Supported: formats={self.supported_formats}, codecs={self.supported_codecs}, "
                f"rates={self.supported_sample_rates}, bits={self.supported_bit_rates}, "
                f"channels={self.supported_channels}"
            )
            return SpeechResult(None, SpeechResultState.ERROR)

        app_id = self._config[CONF_APP_ID]
        access_token = self._config[CONF_ACCESS_TOKEN]
        resource_id = self._config[CONF_RESOURCE_ID]
        service_url = self._config.get(CONF_SERVICE_URL, DEFAULT_SERVICE_URL)
        self._connect_id = str(uuid.uuid4())

        custom_headers = {
            "X-Api-App-Key": app_id,
            "X-Api-Access-Key": access_token,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": self._connect_id,
        }
        volc_audio_params = {
            "format": self._config.get(CONF_AUDIO_FORMAT, DEFAULT_AUDIO_FORMAT),
            "rate": self._config.get(CONF_AUDIO_RATE, DEFAULT_AUDIO_RATE),
            "bits": self._config.get(CONF_AUDIO_BITS, DEFAULT_AUDIO_BITS),
            "channel": self._config.get(CONF_AUDIO_CHANNEL, DEFAULT_AUDIO_CHANNEL),
            "codec": "raw",
        }
        
        # 获取VAD配置
        end_window_size = self._config.get(CONF_END_WINDOW_SIZE, DEFAULT_END_WINDOW_SIZE)
        force_to_speech_time = self._config.get(CONF_FORCE_TO_SPEECH_TIME, DEFAULT_FORCE_TO_SPEECH_TIME)
        
        # 创建VAD配置
        vad_config = {
            "vad_enable": True,
            "end_window_size": end_window_size,  # 检测非语音部分的窗口大小(毫秒)
        }
        
        # 如果设置了强制识别时间，添加到VAD配置
        if force_to_speech_time > 0:
            vad_config["force_to_speech_time"] = force_to_speech_time
            
        self._perf_log(LOG_TAG_VAD, f"VAD配置: {vad_config}")
        
        request_params = {
            "model_name": "bigmodel",
            "language": self._config.get(CONF_LANGUAGE, DEFAULT_LANGUAGE),
            "enable_itn": self._config.get(CONF_ENABLE_ITN, DEFAULT_ENABLE_ITN),
            "enable_punc": self._config.get(CONF_ENABLE_PUNC, DEFAULT_ENABLE_PUNC),
            "result_type": self._config.get(CONF_RESULT_TYPE, DEFAULT_RESULT_TYPE),
            "show_utterances": self._config.get(CONF_SHOW_UTTERANCES, DEFAULT_SHOW_UTTERANCES),
            "vad": vad_config,  # 添加VAD配置到请求参数中
        }
        full_client_request_payload = {
            "user": {"uid": "homeassistant_user"},
            "audio": volc_audio_params,
            "request": request_params,
        }

        session = async_get_clientsession(self.hass)
        # 使用集合跟踪已处理文本，避免重复
        processed_text_set = set()
        # 存储所有接收到的文本片段，按接收顺序
        all_text_segments = []
        # 存储最终结果
        final_results = []
        server_marked_final = False
        error_occurred = False
        error_payload_for_logging = None
        # 获取是否只在文本变化时记录日志
        log_text_change_only = self._config.get(CONF_LOG_TEXT_CHANGE_ONLY, DEFAULT_LOG_TEXT_CHANGE_ONLY)
        # 跟踪上一个识别文本，用于比较变化
        last_recognized_text = ""
        
        try:
            ws_connect_start = time.time()
            self._perf_log(LOG_TAG_WEBSOCKET, f"开始连接WebSocket: {service_url}")
            
            async with session.ws_connect(service_url, headers=custom_headers) as websocket:
                ws_connect_time = time.time() - ws_connect_start
                self._perf_log(LOG_TAG_WEBSOCKET, f"WebSocket连接成功，耗时: {ws_connect_time:.3f}秒")
                
                _LOGGER.info(f"Connected to Volcengine ASR: {service_url} with connect_id: {self._connect_id}")
                
                # 发送初始请求参数
                payload_send_start = time.time()
                payload_json_bytes = json.dumps(full_client_request_payload).encode("utf-8")
                msg_header = struct.pack(">BBBB", 0x11, 0x10, 0x10, 0x00)
                payload_size = struct.pack(">I", len(payload_json_bytes))
                await websocket.send_bytes(msg_header + payload_size + payload_json_bytes)
                
                payload_send_time = time.time() - payload_send_start
                self._perf_log(LOG_TAG_WEBSOCKET, f"初始请求参数发送完成，耗时: {payload_send_time:.3f}秒")
                _LOGGER.debug(f"Sent Full Client Request: {full_client_request_payload}")

                # Loop to send audio and receive intermediate results
                audio_chunks_batch = []
                performance_mode = self._config.get(CONF_PERFORMANCE_MODE, DEFAULT_PERFORMANCE_MODE)
                batch_size = PERF_AUDIO_BATCH_SIZE if performance_mode else 1
                timeout_send = PERF_RESPONSE_TIMEOUT_SEND if performance_mode else 0.5
                timeout_final = PERF_RESPONSE_TIMEOUT_FINAL if performance_mode else 10.0
                
                self._perf_log("CONFIG", f"性能模式: {performance_mode}, 批量大小: {batch_size}, 发送超时: {timeout_send}, 最终超时: {timeout_final}")
                
                # 音频流处理开始
                audio_stream_start = time.time()
                self._perf_log("AUDIO_STREAM", "开始处理音频流")
                
                async for audio_chunk in stream:
                    if not audio_chunk or error_occurred:
                        continue
                    
                    # 添加到批次
                    audio_chunks_batch.append(audio_chunk)
                    
                    # 如果达到最大批次大小，则发送
                    if len(audio_chunks_batch) >= batch_size:
                        send_start = time.time()
                        await self._send_audio_chunks(websocket, audio_chunks_batch, False)
                        send_time = time.time() - send_start
                        
                        chunk_total_size = sum(len(chunk) for chunk in audio_chunks_batch)
                        self._audio_chunks_sent += len(audio_chunks_batch)
                        self._audio_bytes_sent += chunk_total_size
                        
                        if self._first_audio_send_time == 0:
                            self._first_audio_send_time = time.time()
                        self._last_audio_send_time = time.time()
                        
                        self._perf_log(LOG_TAG_AUDIO_SEND, 
                            f"发送音频块 {self._audio_chunks_sent}个, 共{chunk_total_size}字节, 耗时: {send_time:.3f}秒")
                        
                        audio_chunks_batch = []
                        
                        # 尝试接收响应，但不等待太久
                        recv_start = time.time()
                        await self._try_receive_responses(websocket, processed_text_set, all_text_segments, 
                                                        final_results, error_occurred, server_marked_final, 
                                                        error_payload_for_logging, last_recognized_text, 
                                                        log_text_change_only, timeout_send)
                        recv_time = time.time() - recv_start
                        
                        self._perf_log(LOG_TAG_RESPONSE_RECEIVE, f"接收响应尝试耗时: {recv_time:.3f}秒")
                        
                        # 更新上一次识别文本
                        if all_text_segments and all_text_segments[-1]["text"]:
                            last_recognized_text = all_text_segments[-1]["text"]
                
                audio_stream_time = time.time() - audio_stream_start
                self._perf_log("AUDIO_STREAM", f"音频流处理完成，总耗时: {audio_stream_time:.3f}秒")
                
                # 发送剩余的音频块
                if audio_chunks_batch and not error_occurred:
                    send_start = time.time()
                    await self._send_audio_chunks(websocket, audio_chunks_batch, False)
                    send_time = time.time() - send_start
                    
                    chunk_total_size = sum(len(chunk) for chunk in audio_chunks_batch)
                    self._audio_chunks_sent += len(audio_chunks_batch)
                    self._audio_bytes_sent += chunk_total_size
                    
                    self._perf_log(LOG_TAG_AUDIO_SEND, 
                        f"发送剩余音频块 {len(audio_chunks_batch)}个, 共{chunk_total_size}字节, 耗时: {send_time:.3f}秒")
                    
                    audio_chunks_batch = []
                
                # Send final empty audio chunk if no error occurred during streaming
                if not error_occurred:
                    final_send_start = time.time()
                    _LOGGER.debug("Finished audio stream. Sending final empty chunk.")
                    flags = 0b0010  # Mark as final chunk
                    msg_type_flags = (0b0010 << 4) | flags
                    final_audio_header = struct.pack(">BBBB", 0x11, msg_type_flags, 0x00, 0x00)
                    final_audio_payload_size = struct.pack(">I", 0)
                    await websocket.send_bytes(final_audio_header + final_audio_payload_size)
                    final_send_time = time.time() - final_send_start
                    
                    self._perf_log(LOG_TAG_AUDIO_SEND, f"发送最终标记（空音频块）耗时: {final_send_time:.3f}秒")
                    _LOGGER.debug("Sent final empty audio chunk.")
                else:
                    _LOGGER.warning(f"Skipping final empty chunk due to earlier error. Error details: {error_payload_for_logging}")

                # Wait for final ASR responses
                final_recv_start = time.time()
                self._perf_log(LOG_TAG_RESPONSE_RECEIVE, f"等待最终响应，超时: {timeout_final}秒")
                
                # 添加文本提取时间检查
                if self._first_text_time > 0 and not server_marked_final:
                    text_extraction_time = time.time() - self._first_text_time
                    # 如果从第一次提取文本已经过去了2秒以上，并且有至少一个文本段，可以考虑提前结束
                    if text_extraction_time > 2.0 and all_text_segments and len(all_text_segments) > 2:
                        self._perf_log(LOG_TAG_RESPONSE_RECEIVE, f"从首次提取文本已经过去 {text_extraction_time:.3f} 秒，可能可以提前结束")
                        # 如果最后一次文本提取已经超过1.5秒没有变化，就提前结束
                        if time.time() - self._last_resp_time > 1.5:
                            self._perf_log(LOG_TAG_RESPONSE_RECEIVE, "文本已稳定，提前结束等待")
                            server_marked_final = True
                
                await self._try_receive_responses(websocket, processed_text_set, all_text_segments, 
                                               final_results, error_occurred, server_marked_final, 
                                               error_payload_for_logging, last_recognized_text,
                                               log_text_change_only, timeout_final)
                
                final_recv_time = time.time() - final_recv_start
                self._perf_log(LOG_TAG_RESPONSE_RECEIVE, f"接收最终响应耗时: {final_recv_time:.3f}秒")
                
                # 构建最终结果
                final_text = ""
                
                # 优先使用服务器标记的final结果
                if final_results:
                    # 如果有多个final结果，选择最长的一个
                    final_text = max(final_results, key=len)
                    _LOGGER.info(f"Using final result from server: \"{final_text}\"")
                # 如果没有final结果，但有其他文本片段
                elif all_text_segments:
                    # 简化决策逻辑：直接使用最后一个文本片段（通常是最完整的）
                    final_text = all_text_segments[-1]["text"]
                    _LOGGER.info(f"Using latest text segment: \"{final_text}\"")
                
                # 输出性能统计
                total_process_time = time.time() - self._process_start_time
                audio_send_time = self._last_audio_send_time - self._first_audio_send_time if self._first_audio_send_time > 0 else 0
                resp_time = self._last_resp_time - self._first_resp_time if self._first_resp_time > 0 else 0
                
                self._perf_log("STATS", f"""
处理统计:
- 总处理时间: {total_process_time:.3f}秒
- 音频发送时间段: {audio_send_time:.3f}秒
- 响应接收时间段: {resp_time:.3f}秒
- 发送的音频块数: {self._audio_chunks_sent}
- 发送的音频字节数: {self._audio_bytes_sent}
- 接收的响应数: {self._responses_received}
- 文本提取次数: {self._text_extraction_count}
- 首次文本提取时间: {self._first_text_time - self._process_start_time:.3f}秒
                """)
                
                _LOGGER.info(f"Final recognized text: \"{final_text}\"")

                if final_text: # 如果我们有任何文本，则为成功
                    _LOGGER.info(f"ASR process finished with text. Error state: {error_occurred}, Server final: {server_marked_final}")
                    return SpeechResult(final_text, SpeechResultState.SUCCESS)
                elif server_marked_final and not error_occurred:
                    # 服务器标记了最终状态，但没有文本（可能是静音识别）
                    _LOGGER.info("ASR process ended: Server marked final with no text. Likely silence. Returning SUCCESS with no text.")
                    return SpeechResult("", SpeechResultState.SUCCESS)
                elif error_occurred: # 没有文本，发生了错误
                    _LOGGER.error(f"ASR process ended with an error and no recognized text. Details: {error_payload_for_logging}")
                    return SpeechResult(None, SpeechResultState.ERROR)
                else: # 没有文本，没有明确的错误（例如静音，或服务器在没有最终消息的情况下关闭）
                    _LOGGER.warning("ASR process ended with no recognized text and no explicit error. Returning ERROR state.")
                    return SpeechResult(None, SpeechResultState.ERROR)

        except aiohttp.ClientError as e:
            _LOGGER.error(f"aiohttp client connection error: {e}", exc_info=True)
            return SpeechResult(None, SpeechResultState.ERROR)
        except Exception as e:
            _LOGGER.error(f"An unexpected error occurred in Volcengine ASR processing: {e}", exc_info=True)
            return SpeechResult(None, SpeechResultState.ERROR)

    async def _send_audio_chunks(self, websocket, chunks, is_final=False):
        """发送一批音频块"""
        for i, audio_chunk in enumerate(chunks):
            # 只有最后一个块在需要时标记为final
            is_last_and_final = is_final and i == len(chunks) - 1
            flags = 0b0010 if is_last_and_final else 0b0000
            msg_type_flags = (0b0010 << 4) | flags
            audio_header = struct.pack(">BBBB", 0x11, msg_type_flags, 0x00, 0x00)
            audio_payload_size = struct.pack(">I", len(audio_chunk))
            
            chunk_send_start = time.time()
            await websocket.send_bytes(audio_header + audio_payload_size + audio_chunk)
            chunk_send_time = time.time() - chunk_send_start
            
            if self._enable_perf_log and (i == 0 or i == len(chunks) - 1):
                self._perf_log(LOG_TAG_AUDIO_SEND, 
                    f"音频块 #{i+1}/{len(chunks)}, 大小: {len(audio_chunk)}字节, 发送耗时: {chunk_send_time:.3f}秒, is_final: {is_last_and_final}")
            
            _LOGGER.debug(f"Sent audio chunk, size: {len(audio_chunk)}, is_final: {is_last_and_final}")
    
    async def _try_receive_responses(self, websocket, processed_text_set, all_text_segments, 
                                   final_results, error_occurred, server_marked_final, 
                                   error_payload_for_logging, last_recognized_text,
                                   log_text_change_only, timeout=0.1):
        """尝试接收并处理响应，但有超时限制"""
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + timeout
        
        self._perf_log(LOG_TAG_RESPONSE_RECEIVE, f"开始接收响应，超时: {timeout}秒")
        recv_count = 0
        
        while asyncio.get_event_loop().time() < end_time and not server_marked_final and not error_occurred:
            try:
                remaining_time = end_time - asyncio.get_event_loop().time()
                if remaining_time <= 0:
                    break
                
                ws_msg_receive_start = time.time()
                ws_msg = await asyncio.wait_for(websocket.receive(), timeout=remaining_time)
                ws_msg_receive_time = time.time() - ws_msg_receive_start
                recv_count += 1
                
                if self._first_resp_time == 0:
                    self._first_resp_time = time.time()
                self._last_resp_time = time.time()
                self._responses_received += 1
                
                self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
                    f"收到WS消息 #{recv_count}, 类型: {ws_msg.type}, 接收耗时: {ws_msg_receive_time:.3f}秒")
                
                if ws_msg.type == aiohttp.WSMsgType.BINARY:
                    response_data = ws_msg.data
                    if not response_data or len(response_data) < 8:
                        _LOGGER.debug("ASR: Empty/incomplete binary msg, skipping.")
                        continue
                    
                    resp_header_data = response_data[:4]
                    raw_payload_data = response_data[12:]
                    
                    preprocess_start = time.time()
                    processed_payload_data = self._preprocess_payload(raw_payload_data)
                    preprocess_time = time.time() - preprocess_start
                    
                    self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
                        f"预处理响应数据耗时: {preprocess_time:.3f}秒, 原始大小: {len(raw_payload_data)}字节, 处理后大小: {len(processed_payload_data)}字节")
                    
                    resp_msg_type = (resp_header_data[1] >> 4) & 0x0F
                    
                    if resp_msg_type == 0b1001:  # Server ASR Result
                        if not processed_payload_data:
                            _LOGGER.debug("ASR: Empty payload after preprocess, skipping.")
                            continue
                        try:
                            decode_start = time.time()
                            decoded_payload = processed_payload_data.decode("utf-8")
                            resp_json = json.loads(decoded_payload)
                            decode_time = time.time() - decode_start
                            
                            self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
                                f"解码JSON响应耗时: {decode_time:.3f}秒, JSON大小: {len(decoded_payload)}字节")
                            
                            # 只在文本变化时记录日志或不启用此功能时始终记录
                            has_text_changed = False
                            
                            msg_type = resp_json.get("type")
                            msg_status = resp_json.get("header", {}).get("status", 0)
                            
                            self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
                                f"响应类型: {msg_type}, 状态码: {msg_status}")
                            
                            if msg_type == "error" or (msg_status != 20000000 and msg_status != 0 and msg_type == "final"):
                                _LOGGER.error(f"Volcengine ASR Error in payload: {resp_json}")
                                error_payload_for_logging = resp_json
                                error_occurred = True
                                break
                            
                            # 提取识别结果文本
                            extract_start = time.time()
                            result_data = resp_json.get("result")
                            extracted_texts = []
                            
                            if isinstance(result_data, list):
                                for res_item in result_data:
                                    if isinstance(res_item, dict):
                                        text = res_item.get("text", "")
                                        if text and text.strip():
                                            extracted_texts.append(text.strip())
                                    elif isinstance(res_item, str) and res_item.strip():
                                        extracted_texts.append(res_item.strip())
                            elif isinstance(result_data, dict):
                                text = result_data.get("text", "")
                                if text and text.strip():
                                    extracted_texts.append(text.strip())
                            elif isinstance(result_data, str) and result_data.strip():
                                extracted_texts.append(result_data.strip())
                            
                            extract_time = time.time() - extract_start
                            
                            self._perf_log(LOG_TAG_TEXT_EXTRACT, 
                                f"文本提取耗时: {extract_time:.3f}秒, 提取文本数: {len(extracted_texts)}")
                            
                            # 处理提取的文本
                            if extracted_texts:
                                self._text_extraction_count += 1
                                if self._first_text_time == 0:
                                    self._first_text_time = time.time()
                                    self._perf_log(LOG_TAG_TEXT_EXTRACT, 
                                        f"首次文本提取时间: +{self._first_text_time - self._process_start_time:.3f}秒")
                            
                            for text in extracted_texts:
                                if text not in processed_text_set:
                                    processed_text_set.add(text)
                                    all_text_segments.append({"text": text, "is_final": msg_type == "final"})
                                    
                                    # 检查文本是否变化
                                    if text != last_recognized_text:
                                        has_text_changed = True
                                        self._perf_log(LOG_TAG_TEXT_EXTRACT, 
                                            f"文本已更改: \"{text}\"")
                                        # 文本有变化时更新最后响应时间
                                        self._last_resp_time = time.time()
                            
                            # 根据文本变化情况决定是否记录日志
                            if not log_text_change_only or has_text_changed or msg_type == "final":
                                _LOGGER.debug(f"ASR Response: {resp_json}")
                            
                            # 如果是最终结果，特殊标记
                            if msg_type == "final":
                                self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
                                    f"收到最终响应标记，提取文本数: {len(extracted_texts)}")
                                    
                                for text in extracted_texts:
                                    if text:  # 确保有内容
                                        final_results.append(text)
                                server_marked_final = True
                                return True  # 收到最终结果后立即返回
                                
                        except UnicodeDecodeError as ude_err:
                            _LOGGER.warning(f"ASR: UnicodeDecodeError: {ude_err}. Payload (hex): {processed_payload_data.hex()}")
                        except json.JSONDecodeError as json_err:
                            _LOGGER.warning(f"ASR: JSONDecodeError: {json_err}. Original (hex): {raw_payload_data.hex()}, Processed: {processed_payload_data.decode('utf-8', errors='ignore')}")
                        except AttributeError as attr_err:
                            _LOGGER.error(f"ASR: AttributeError processing result: {attr_err}. Response JSON: {resp_json}", exc_info=True)
                            error_payload_for_logging = resp_json
                            error_occurred = True
                            break
                    
                    elif resp_msg_type == 0b1111:  # Server Error Message
                        err_msg = processed_payload_data.decode('utf-8', errors='ignore')
                        _LOGGER.error(f"Volcengine ASR WebSocket Error Message: {err_msg}")
                        error_payload_for_logging = err_msg
                        error_occurred = True
                        break
                        
                elif ws_msg.type == aiohttp.WSMsgType.ERROR:
                    _LOGGER.error(f"aiohttp WS Error: {websocket.exception()}")
                    error_occurred = True
                    break
                elif ws_msg.type == aiohttp.WSMsgType.CLOSED:
                    _LOGGER.info("aiohttp WS Closed by server.")
                    # 确保处理已接收的文本
                    if all_text_segments and not final_results:
                        server_marked_final = True
                    self._perf_log(LOG_TAG_WEBSOCKET, "服务器关闭WebSocket连接，标记为结束")
                    break
                    
            except asyncio.TimeoutError:
                # 超时是正常的，继续检查时间
                pass
            except Exception as e:
                _LOGGER.error(f"ASR: Unexpected error processing response: {e}", exc_info=True)
                error_occurred = True
                break
        
        self._perf_log(LOG_TAG_RESPONSE_RECEIVE, 
            f"接收响应循环结束，收到 {recv_count} 条消息，是否最终标记: {server_marked_final}, 是否出错: {error_occurred}")
        return server_marked_final

