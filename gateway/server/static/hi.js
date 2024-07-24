function makeBlackPixelsTransparent(maskCanvas) {
    let maskCtx = maskCanvas.getContext('2d');
    let idata = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    // 0,0,0,255 -> 0,0,0,0 (make it transparent from color black)
    let bytes = new Uint8ClampedArray(idata.data.buffer);
    for (let i=0; i<bytes.length; i+=4) {
        if (bytes[i+0] == 0) {
            bytes[i+3] = 0;
        }
    }
    maskCtx.putImageData(idata, 0, 0);
    return maskCanvas;
}

async function getCanvasForImage(imgSrc) {
    return new Promise((resolve, reject) => {
        const canvas = document.createElement('canvas');
        const image = new Image();
        image.onload = () => {
            canvas.width = image.width;
            canvas.height = image.height;
            canvas.getContext("2d").drawImage(image, 0, 0);
            resolve(canvas);
        };
        image.src = imgSrc;
    });
}

function getCutoutCanvas(imgCanvas, maskCanvas) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext("2d");
    canvas.height = imgCanvas.height;
    canvas.width = imgCanvas.width;
    
    ctx.drawImage(imgCanvas, 0, 0);
    ctx.globalCompositeOperation = "destination-in";
    ctx.drawImage(maskCanvas, 0, 0);

    return canvas;
}

async function drawPitchers(imagesrc, json) {
    let canvas = document.getElementById("segmentedPitchersCanvas");
    let ctx = canvas.getContext("2d");

    const imgCanvas = await getCanvasForImage(imagesrc);    
    const maskCanvases = []
    for (let i=0; i<json.length; i++) {
        let maskCanvas = await getCanvasForImage("data:image/png;base64," + json[i]['base64png'])
        maskCanvas = makeBlackPixelsTransparent(maskCanvas);
        maskCanvases.push(maskCanvas);
    }

    canvas.height = imgCanvas.height;
    canvas.width = imgCanvas.width;
    for (const maskCanvas of maskCanvases) {
        const res = getCutoutCanvas(imgCanvas, maskCanvas);
        ctx.drawImage(res, 0, 0);
    }
}

async function drawInUnsegmentedPitcher(imageSrc) {
    const canv = await getCanvasForImage(imageSrc);
    const pageCanv = document.getElementById("unsegmentedPitchersCanvas");
    pageCanv.width = canv.width;
    pageCanv.height = canv.height;
    pageCanv.getContext("2d").drawImage(canv, 0, 0);
}

async function makeInference(imageSrc, formData) {
    drawInUnsegmentedPitcher(imageSrc);
    try {
        const response = await fetch('https://nwws.cc/predict', {
            method: 'POST',
            body: formData
        });
        let data = await response.json();
        await drawPitchers(imageSrc, data);

        // Remove long base64png's from text
        for (let i=0; i<data.length; i++)
            delete data[i]['base64png'];
        document.getElementById('response').textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('response').textContent = 'Error uploading file.';
    }

}

window.onload = () => {
    document.getElementById('uploadForm').addEventListener('submit', async (event) => {
        event.preventDefault();

        const fileInput = document.getElementById('file');
        const file = fileInput.files[0];
        document.getElementById('response').textContent = "Loading...";

        const reader = new FileReader();
        reader.onload = function (e) {
            const imagesrc = e.target.result;
            const formData = new FormData();
            formData.append('file', file);
            makeInference(imagesrc, formData);
        };
        reader.readAsDataURL(file);
    });
}