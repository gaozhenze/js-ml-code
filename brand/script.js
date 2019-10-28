import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';
import { img2x } from './utils';

const MOBILENET_MODEL_PATH = 'http://127.0.0.1:8080/mobilenet/web_model/model.json';
const NUM_CLASSES = 3;

window.onload = async () => {
    const { inputs, labels } = await getInputs();
    const surface = tfvis.visor().surface({ name: '输入示例', styles: { height: 250 } });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });

    const mobilenet = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    mobilenet.summary();
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    const truncatedMobilenet = tf.model({
        inputs: mobilenet.inputs,
        outputs: layer.output
    });

    const model = tf.sequential();
    model.add(tf.layers.flatten({
        inputShape: layer.outputShape.slice(1)
    }));
    model.add(tf.layers.dense({
        units: 10,
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax'
    }));
    model.compile({ loss: 'categoricalCrossentropy', optimizer: tf.train.adam() });

    const { xs, ys } = tf.tidy(() => {
        const xs = tf.concat(inputs.map(imgEl => truncatedMobilenet.predict(img2x(imgEl))));
        const ys = tf.tensor(labels);
        return { xs, ys };
    });

    await model.fit(xs, ys, {
        epochs: 20,
        callbacks: tfvis.show.fitCallbacks(
            { name: '训练效果' },
            ['loss'],
            { callbacks: ['onEpochEnd'] }
        )
    });
};