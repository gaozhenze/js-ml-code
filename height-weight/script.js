import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

window.onload = () => {
    const heights = [150, 160, 170];
    const weights = [40, 50, 60];

    tfvis.render.scatterplot(
        { name: '身高体重训练数据' },
        { values: heights.map((x, i) => ({ x, y: weights[i] })) },
        {
            xAxisDomain: [140, 180],
            yAxisDomain: [30, 70]
        }
    );

    const inputs = tf.tensor(heights).sub(150).div(20);
    inputs.print();
    const labels = tf.tensor(weights).sub(40).div(20);
    labels.print();
};