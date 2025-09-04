import asyncio
import os
import pyaudio
import wave
import oci
from oci.config import from_file
from oci_ai_speech_realtime import (
    RealtimeSpeechClient,
    RealtimeSpeechClientListener,
    RealtimeParameters,
)
from oci.generative_ai_inference import GenerativeAiInferenceClient
from langchain_community.chat_models import ChatOCIGenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

# Variables globales
queue = asyncio.Queue()
cummulativeResult = ""
last_result_time = None
latest_thetime = None
latest_question = None
latest_answer = None
is_playing_audio = False  # FLAG PARA EVITAR BUCLE DE RETROALIMENTACIÓN
audio_start_time = None

def audio_callback(in_data, frame_count, time_info, status):
    # NO PROCESAR AUDIO DEL MICRÓFONO MIENTRAS SE REPRODUCE LA RESPUESTA
    if not is_playing_audio:
        queue.put_nowait(in_data)
    return (None, pyaudio.paContinue)

SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
BUFFER_DURATION_MS = 96
FRAMES_PER_BUFFER = int(SAMPLE_RATE * BUFFER_DURATION_MS / 1000)

p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=FRAMES_PER_BUFFER,
    stream_callback=audio_callback,
)
stream.start_stream()

# Configuración del LLM con parámetros mejorados
compartment_id = os.getenv('OCID')
config = from_file(".oci/config2", "DEFAULT")

llm = ChatOCIGenAI(
    model_id="ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyaodm6rdyxmdzlddweh4amobzoo4fatlao2pwnekexmosq",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    provider="cohere",
    compartment_id=compartment_id,
    model_kwargs={
        "temperature": 0.7,
        "max_tokens": 300,  # Reducir para respuestas más concisas
        "top_p": 0.9,
        "top_k": 50
    },
    auth_type="API_KEY",
    auth_profile="DEFAULT",
    auth_file_location=".oci/config",
)

class SpeechListener(RealtimeSpeechClientListener):
    def on_result(self, result):
        global cummulativeResult, last_result_time, is_playing_audio
        
        # IGNORAR TRANSCRIPCIONES MIENTRAS SE REPRODUCE AUDIO
        if is_playing_audio:
            print(f"[IGNORADO - REPRODUCIENDO AUDIO] {result['transcriptions'][0]['transcription']}")
            return
            
        if result["transcriptions"][0]["isFinal"]:
            transcription = result['transcriptions'][0]['transcription']
            
            # FILTRAR TRANSCRIPCIONES QUE PODRÍAN SER ECO DEL AUDIO REPRODUCIDO
            if should_ignore_transcription(transcription):
                print(f"[IGNORADO - POSIBLE ECO] {transcription}")
                return
                
            cummulativeResult += transcription
            print(f"Received final results: {transcription}")
            print(f"Current cummulative result: {cummulativeResult}")                
            last_result_time = asyncio.get_event_loop().time()
        else:
            if not is_playing_audio:  # Solo mostrar parciales si no estamos reproduciendo
                print(f"Received partial results: {result['transcriptions'][0]['transcription']}")

    def on_ack_message(self, ackmessage):
        print(f"ACK received: {ackmessage}")

    def on_connect(self):
        print("Connected to Realtime Speech Service.")

    def on_connect_message(self, connectmessage):
        print(f"Connect message: {connectmessage}")

    def on_network_event(self, ackmessage):
        print(f"Network event: {ackmessage}")

    def on_error(self, exception):
        print(f"An error occurred: {exception}")

def should_ignore_transcription(transcription):
    """Filtrar transcripciones que podrían ser eco del audio reproducido"""
    transcription_lower = transcription.lower().strip()
    
    # Palabras comunes que aparecen en las respuestas del asistente
    assistant_phrases = [
        "soy command",
        "asistente de lenguaje",
        "estoy aquí para ayudar",
        "en qué puedo ayudar",
        "crédito bancario",
        "forma de préstamo",
        "entidad financiera",
        "inteligencia artificial"
    ]
    
    for phrase in assistant_phrases:
        if phrase in transcription_lower:
            return True
    
    # Ignorar transcripciones muy cortas que podrían ser ruido
    if len(transcription_lower) < 3:
        return True
        
    return False

def convert_pcm_to_wav(pcm_file, wav_file, sample_rate=24000, channels=1, sample_width=2):
    """Convierte archivo PCM a WAV para reproducción"""
    try:
        with open(pcm_file, 'rb') as pcm_data:
            pcm_content = pcm_data.read()
        
        with wave.open(wav_file, 'wb') as wav_file_obj:
            wav_file_obj.setnchannels(channels)
            wav_file_obj.setsampwidth(sample_width)
            wav_file_obj.setframerate(sample_rate)
            wav_file_obj.writeframes(pcm_content)
        
        print(f"Converted {pcm_file} to {wav_file}")
        return wav_file
        
    except Exception as e:
        print(f"Error converting PCM to WAV: {e}")
        return None

def play_audio(file_path):
    """Reproduce el archivo de audio generado con control de retroalimentación"""
    global is_playing_audio, audio_start_time
    
    try:
        # ACTIVAR FLAG PARA EVITAR CAPTURA DE AUDIO PROPIO
        is_playing_audio = True
        audio_start_time = time.time()
        
        print("🔊 INICIANDO REPRODUCCIÓN - Micrófono pausado")
        
        wf = wave.open(file_path, 'rb')
        p_audio = pyaudio.PyAudio()
        audio_stream = p_audio.open(
            format=p_audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        data = wf.readframes(1024)
        while data:
            audio_stream.write(data)
            data = wf.readframes(1024)

        audio_stream.stop_stream()
        audio_stream.close()
        p_audio.terminate()
        wf.close()
        
        print("🔊 REPRODUCCIÓN TERMINADA")
        
        # ESPERAR UN POCO MÁS PARA ASEGURAR QUE NO HAY ECO
        await_time = 2  # 2 segundos de espera adicional
        print(f"⏳ Esperando {await_time}s adicionales para evitar eco...")
        time.sleep(await_time)
        
    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        # REACTIVAR MICRÓFONO
        is_playing_audio = False
        print("🎤 MICRÓFONO REACTIVADO - Listo para nueva consulta")

def generate_llm_response(prompt):
    """Genera respuesta usando el LLM configurado"""
    global latest_thetime, latest_question, latest_answer
    
    try:
        print(f"Enviando prompt al LLM: {prompt}")
        
        # Prompt optimizado para respuestas más concisas
        enhanced_prompt = f"""
        Responde de manera concisa y natural en español. 
        Máximo 2 frases cortas.
        
        Pregunta: {prompt}
        """
        
        messages = [HumanMessage(content=enhanced_prompt)]
        response = llm.invoke(messages)
        
        if hasattr(response, 'content'):
            text_response = response.content
        else:
            text_response = str(response)
        
        text_response = text_response.strip()
        print(f"Respuesta completa del LLM: {text_response}")
        
        # Actualizar variables globales
        latest_thetime = datetime.now()
        latest_question = prompt
        latest_answer = text_response
        
        return text_response
        
    except Exception as e:
        print(f"Error generando respuesta del LLM: {e}")
        return f"Lo siento, hubo un error."

def text_to_speech(text):
    """Convierte texto a audio usando Oracle TTS"""
    try:
        # Limitar longitud del texto para TTS
        if len(text) > 500:
            text = text[:500] + "..."
        
        print(f"Convirtiendo a audio: {text}")
        
        ai_speech_client = oci.ai_speech.AIServiceSpeechClient(config)
        
        synthesize_speech_response = ai_speech_client.synthesize_speech(
            synthesize_speech_details=oci.ai_speech.models.SynthesizeSpeechDetails(
                text=text,
                is_stream_enabled=True,
                compartment_id=compartment_id,
                configuration=oci.ai_speech.models.TtsOracleConfiguration(
                    model_family="ORACLE",
                    model_details=oci.ai_speech.models.TtsOracleTts2NaturalModelDetails(
                        model_name="TTS_2_NATURAL",
                        language_code="es-ES",
                        voice_id="Lucas"),
                    speech_settings=oci.ai_speech.models.TtsOracleSpeechSettings(
                        text_type="TEXT",
                        sample_rate_in_hz=24000,
                        output_format="PCM",
                        speech_mark_types=["WORD"])),
                audio_config=oci.ai_speech.models.TtsBaseAudioConfig(
                    config_type="BASE_AUDIO_CONFIG")
            )
        )
        
        pcm_file = "llm_response.pcm"
        wav_file = "llm_response.wav"
        
        with open(pcm_file, "wb") as audio_file:
            audio_file.write(synthesize_speech_response.data.content)
        
        print(f"Audio PCM guardado como {pcm_file}")
        wav_output = convert_pcm_to_wav(pcm_file, wav_file)
        
        return wav_output if wav_output else pcm_file
        
    except Exception as e:
        print(f"Error en text-to-speech: {e}")
        return None

def process_llm_query():
    """Procesa la consulta usando el LLM y genera respuesta en audio"""
    global cummulativeResult
    
    if not cummulativeResult.strip():
        return
    
    print(f"Procesando consulta: {cummulativeResult}")
    
    # Generar respuesta con el LLM
    llm_response = generate_llm_response(cummulativeResult)
    
    if not llm_response or llm_response.strip() == "":
        print("No se recibió respuesta válida del LLM")
        return
    
    # Convertir respuesta a audio
    audio_file = text_to_speech(llm_response)
    
    # Reproducir audio si se generó correctamente
    if audio_file:
        try:
            play_audio(audio_file)
        except Exception as e:
            print(f"Error reproduciendo audio: {e}")
    
    # Limpiar el resultado acumulativo
    cummulativeResult = ""

async def send_audio(client):
    while True:
        try:
            data = await queue.get()
            await client.send_data(data)
        except Exception as e:
            print(f"Error enviando audio: {e}")

async def check_idle():
    global last_result_time, cummulativeResult, is_playing_audio
    while True:
        try:
            if (not is_playing_audio and 
                last_result_time and 
                (asyncio.get_event_loop().time() - last_result_time > 3)):  # Aumentar tiempo a 3 segundos
                
                print("Han pasado 3 segundos desde la última transcripción")
                print(f"Comando detectado: {cummulativeResult}")
                
                if cummulativeResult.strip():
                    process_llm_query()
                
                cummulativeResult = ""
                last_result_time = None
                
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Error en check_idle: {e}")

if __name__ == "__main__":
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        realtime_speech_parameters = RealtimeParameters()
        realtime_speech_parameters.language_code = "es-ES"
        realtime_speech_parameters.model_domain = (
            realtime_speech_parameters.MODEL_DOMAIN_GENERIC
        )
        realtime_speech_parameters.final_silence_threshold_in_ms = 3000  # Aumentar threshold

        realtime_speech_url = "wss://realtime.aiservice.us-phoenix-1.oci.oraclecloud.com"
        speech_client = RealtimeSpeechClient(
            config=config,
            realtime_speech_parameters=realtime_speech_parameters,
            listener=SpeechListener(),
            service_endpoint=realtime_speech_url,
            signer=None,
            compartment_id=compartment_id
        )

        loop.create_task(send_audio(speech_client))
        loop.create_task(check_idle())

        print("🎤 Iniciando sistema de conversación por voz SIN bucle de retroalimentación...")
        print("💡 El micrófono se pausará automáticamente durante la reproducción")
        
        loop.run_until_complete(speech_client.connect())

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error general: {e}")
    finally:
        if stream.is_active():
            stream.close()
        print("✅ Sistema cerrado correctamente")
