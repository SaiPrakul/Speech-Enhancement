let audioRecorder;
let audioBlob;
let audioFile;
let audioURL;

document.getElementById('recordButton').addEventListener('click', startRecording);
document.getElementById('stopButton').addEventListener('click', stopRecording);
document.getElementById('uploadInput').addEventListener('change', handleUpload);
document.getElementById('processButton').addEventListener('click', processAudio);

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioRecorder = new MediaRecorder(stream);
    audioRecorder.ondataavailable = (event) => {
        audioBlob = event.data;
        audioURL = URL.createObjectURL(audioBlob);
        document.getElementById('audioPlayer').src = audioURL;
        document.getElementById('processButton').disabled = false;
    };
    audioRecorder.start();
    document.getElementById('recordButton').disabled = true;
    document.getElementById('stopButton').disabled = false;
    document.getElementById('uploadInput').disabled = true;
}

function stopRecording() {
    audioRecorder.stop();
    document.getElementById('recordButton').disabled = false;
    document.getElementById('stopButton').disabled = true;
    document.getElementById('uploadInput').disabled = false;
}

function handleUpload(event) {
    const file = event.target.files[0];
    if (file) {
        audioBlob = file;
        audioURL = URL.createObjectURL(file);
        document.getElementById('audioPlayer').src = audioURL;
        document.getElementById('processButton').disabled = false;
    }
}

async function processAudio() {
    console.log("Process start")
    const formData = new FormData();
    audioFile = new File([audioBlob], 'audio.wav', { type: 'audio/wav' });
    formData.append('audio', audioFile);

    const response = await fetch('http://localhost:5000/process', {
        method: 'POST',
        body: formData
    });
    console.log(response)
    if(response.ok)
    {
        // console.log(response.)
    const outputBlob = await response.blob();
    const outputURL = URL.createObjectURL(outputBlob);
    console.log(outputURL)
    document.getElementById('outputPlayer').src = outputURL;
    }
    else
    {
        console.error('Processing failed');
    }
}

