import * as tf from '@tensorflow/tfjs';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';

window.onload = async () => {
    const model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
};