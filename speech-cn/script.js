import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer;

window.onload = async () => {
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json'
    );
    await recognizer.ensureModelLoaded();
    transferRecognizer = recognizer.createTransfer('轮播图');
};

window.collect = async (btn) => {
    btn.disabled = true;
    const label = btn.innerText;
    await transferRecognizer.collectExample(
        label === '背景噪音' ? '_background_noise_' : label
    );
    btn.disabled = false;
    document.querySelector('#count').innerHTML = JSON.stringify(transferRecognizer.countExamples(), null, 2);
};