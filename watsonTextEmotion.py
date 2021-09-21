#This project takes the audio from the user and records it 
#It then gives a score on speech clarity based on the text the user meant to say
#It also gives emotion analysis of the spoken words 
__author__ = "Ronak Pai"

from flask import Flask, render_template, request, redirect, Response
import pyaudio
import time
from six.moves import queue
from google.cloud import speech_v1p1beta1 as speech
import stringdist
import string
from ibm_watson import ToneAnalyzerV3
import json


app = Flask(__name__)

text = ""
final_transcript = ""

STREAMING_LIMIT = 10000


def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                                                  * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)

sampleRate = 16000
CHUNK = int(sampleRate / 10)  # 100ms
stream = ResumableMicrophoneStream(sampleRate, CHUNK)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream.result_end_time = int((result_seconds * 1000)
                                     + (result_nanos / 1000000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:
            return transcript
            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

        else:
            stream.last_transcript_was_final = False


@app.route('/audio', methods = ['POST', 'GET'])
def audio():
    global final_transcript
    def sound():
        results = ""
        lastTranscript = ""

        stream.__enter__()

        client = speech.SpeechClient()
        config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sampleRate,
            language_code='en-US',
            max_alternatives=1)
        streaming_config = speech.types.StreamingRecognitionConfig(
            config=config,
            interim_results=True)

        print("recording...")

        t_end = time.time() + 5
        while not stream.closed:
            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content) for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)

            # Now, put the transcription responses to use.
            transcript = listen_print_loop(responses, stream)
            if transcript != None and transcript != lastTranscript:
                lastTranscript = transcript
                results += transcript

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            stream.new_stream = True
        print(results)
        return results
    final_transcript = sound()
    return final_transcript


@app.route('/recieve', methods = ['POST', 'GET'])
def getText():
    #Get the text from the website
    global text
    data = request.get_json(force=True)
    text = str(data[0])
    print(text)
    return "success"


@app.route('/close', methods = ['POST', 'GET'])
def close():
    #Close the audio stream
    stream.closed = True
    return "success"


@app.route('/result', methods = ['POST', 'GET'])
def result():
    #Get the accuracy from combining the text to the heard words  
    global text
    global final_transcript
    data = []
    text = text.lower()
    final_transcript = final_transcript.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    final_transcript = final_transcript.translate(str.maketrans('', '', string.punctuation))
    grade = str(int((1 - stringdist.rdlevenshtein_norm(text, final_transcript))*100)) + "%"
    print(grade)
    return grade


@app.route('/emotion', methods = ['POST', 'GET'])
def emotions():
    #Get the emotions for the text from Watson API
    global text
    emotion = []
    tone_analyzer = ToneAnalyzerV3(
        version='2017-09-21',
        iam_apikey='Iw1X1i5s-OK_c5RRmmWBVGRQyDwoqmtQ-NXvbQVBAcs7',
        url='https://gateway.watsonplatform.net/tone-analyzer/api'
    )
    tone_analysis = tone_analyzer.tone(
        {'text': text},
        content_type='application/json'
    ).get_result()

    print(tone_analysis)
    document = "Overall speech tone: " + str(int(tone_analysis["document_tone"]["tones"][0]["score"]*100)) + "% "
    document += tone_analysis["document_tone"]["tones"][0]["tone_name"]
    emotion.append(document)

    if "sentences_tone" in tone_analysis:
        for sentence in tone_analysis["sentences_tone"]:
            detect = sentence["text"] + ": "
            if len(sentence["tones"]) > 0:
                detect += str(int(sentence["tones"][0]["score"]*100)) + "% "
                detect += sentence["tones"][0]["tone_name"]
            else:
                detect += "No emotion detected"
            emotion.append(detect)
    return json.dumps(emotion)

#Render the website
@app.route('/')
def index():
    return render_template('index.html')

#Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True, port=5000)
