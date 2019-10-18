import * as tfvis from '@tensorflow/tfjs-vis';
import { getData } from './data';

window.onload = () => {
    const data = getData(200, 3);

    tfvis.render.scatterplot(
        { name: '训练数据' },
        {
            values: [
                data.filter(p => p.label === 1),
                data.filter(p => p.label === 0),
            ]
        }
    );
};