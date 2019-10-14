import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data.js';

window.onload = () => {
    const data = getData(400);

    tfvis.render.scatterplot(
        { name: 'XOR 训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );

    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 4,
        inputShape: [2],
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
};