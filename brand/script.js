import * as tfvis from '@tensorflow/tfjs-vis';
import { getInputs } from './data';

window.onload = async () => {
    const { inputs, labels } = await getInputs();
    const surface = tfvis.visor().surface({ name: '输入示例', styles: { height: 250 } });
    inputs.forEach(img => {
        surface.drawArea.appendChild(img);
    });
};