import * as speechCommands from '@tensorflow-models/speech-commands';

const MODEL_PATH = 'http://127.0.0.1:8080';
let transferRecognizer;

window.onload = async () => {
    const recognizer = speechCommands.create(
        'BROWSER_FFT',
        null,
        MODEL_PATH + '/speech/model.json',
        MODEL_PATH + '/speech/metadata.json',
    );
    await recognizer.ensureModelLoaded();
    transferRecognizer = recognizer.createTransfer('轮播图');
    const res = await fetch(MODEL_PATH + '/slider/data.bin');
    const arrayBuffer = await res.arrayBuffer();
    transferRecognizer.loadExamples(arrayBuffer);
    await transferRecognizer.train({ epochs: 30 });
    console.log('done');
};

window.toggle = async (checked) => {
    if (checked) {
        await transferRecognizer.listen(result => {
            const { scores } = result;
            const labels = transferRecognizer.wordLabels();
            const index = scores.indexOf(Math.max(...scores));
            console.log(labels[index]);
        }, {
            overlapFactor: 0,
            probabilityThreshold: 0.75
        });
    } else {
        transferRecognizer.stopListening();
    }
};