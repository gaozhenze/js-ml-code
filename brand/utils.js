import * as tf from '@tensorflow/tfjs';

export function img2x(imgEl){
    return tf.tidy(() => {
        const input = tf.browser.fromPixels(imgEl)
            .toFloat()
            .sub(255 / 2)
            .div(255 / 2)
            .reshape([1, 224, 224, 3]);
        return input;
    });
}