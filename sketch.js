const xCoordinates = [];
const yCoordinates = [];
const width = 1000;
const height = 500;
const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

let m, b;

const loss = (preds, labels) => preds.sub(labels).square().mean();
const predict = (x) => m.mul(x).add(b);
const normaliseX = (x) => map(x, 0, width, 0, 1);
const normaliseY = (y) => map(y, 0, height, 1, 0);
const deNormaliseY = (y) => map(y, 0, 1, height, 0);

function setup() {
    createCanvas(width, height);

    m = tf.variable(tf.scalar(random()));
    b = tf.variable(tf.scalar(random()));
}

function draw() {
    background(0);

    if (xCoordinates.length > 0) {
        minimiseLoss(xCoordinates, yCoordinates);
        drawLine();
    }

    xCoordinates.forEach((x, i) => {
        drawPoint(x, yCoordinates[i]);
    });
}

function minimiseLoss(xs, yLabels) {
    tf.tidy(() => {
        const xs_tensor = tf.tensor1d(xs.map(x => normaliseX(x)));
        const yLabels_tensor = tf.tensor1d(yLabels.map(y => normaliseY(y)));

        optimizer.minimize(() => loss(predict(xs_tensor), yLabels_tensor));
    });
}

function drawLine() {
    const x1 = 0;
    const x2 = width;
    let y1, y2;
    tf.tidy(() => {
        const y1_tensor = predict(tf.scalar(normaliseX(x1)));
        const y2_tensor = predict(tf.scalar(normaliseX(x2)));
        y1 = deNormaliseY(y1_tensor.dataSync()[0]);
        y2 = deNormaliseY(y2_tensor.dataSync()[0]);
    });

    strokeWeight(2);
    stroke(255);
    line(x1, y1, x2, y2);
}

function drawPoint(x, y) {
    strokeWeight(8);
    stroke(255);
    point(x, y);
}

function mouseClicked() {
    xCoordinates.push(mouseX);
    yCoordinates.push(mouseY);
}