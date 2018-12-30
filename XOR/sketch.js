const canvasLength = 800;
const resolution = 10;
const rows = canvasLength / resolution;
const cols = canvasLength / resolution;

let model;

function setup() {
    createCanvas(canvasLength, canvasLength);
    frameRate(30);

    setupInputs();
    buildModel();
    trainModel();
}

function setupInputs() {
    const matrixVals = [];
    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            matrixVals.push([i / rows, j / cols]);
        }
    }

    inputs = tf.tensor2d(matrixVals);
}

function buildModel() {
    const learningRate = 0.1;
    const hiddenLayer = tf.layers.dense({
        units: 14,
        inputShape: [2],
        activation: 'sigmoid'
    });
    const outputLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    });
    model = tf.sequential({
        layers: [hiddenLayer, outputLayer]
    });
    model.compile({
        optimizer: new tf.train.adam(learningRate),
        loss: tf.losses.meanSquaredError
    });
}

function trainModel() {
    const trainingConfig = {
        epochs: 5,
        shuffle: true
    };
    const trainingXs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
    const trainingYs = tf.tensor2d([[0], [1], [1], [0]]);

    train(trainingXs, trainingYs, trainingConfig);
}

function train(trainingXs, trainingYs, trainingConfig) {
    model.fit(trainingXs, trainingYs, trainingConfig)
        .then(res => {
            console.log(`Loss: ${res.history.loss[0]}`);
            setTimeout(() => train(trainingXs, trainingYs, trainingConfig), 20);
        });
}

function predict() {
    let prediction;
    prediction = model.predict(inputs);

    return prediction;
}

async function draw() {
    background(0);

    const prediction = predict();
    const yVals = await prediction.data();
    prediction.dispose();

    let index = 0;
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let brightness = yVals[index] * 255
            noStroke();
            fill(brightness);
            rect(i * resolution, j * resolution, resolution, resolution);
            fill(255 - brightness);
            textSize(8);
            textAlign(CENTER, CENTER);
            text(nf(yVals[index], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2)
            index++;
        }
    }
}
