import { getIrisData, IRIS_CLASSES } from './data';

window.onload = () => {
    const [xTrain, yTrain, xTest, yTest] = getIrisData(0.15);
    console.log(xTrain);
    console.log(yTrain);
    console.log(xTest);
    console.log(yTest);
    console.log(IRIS_CLASSES);
};