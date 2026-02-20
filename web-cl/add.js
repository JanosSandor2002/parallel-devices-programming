async function runWebCL() {
  if (window.WebCL == undefined) {
    document.getElementById('result').innerText =
      'WebCL nem támogatott ebben a böngészőben.';
    return;
  }

  try {
    // 1. Kernel betöltése .cl fájlból
    const response = await fetch('add.cl');
    const kernelSource = await response.text();

    // 2. Platform és device
    var platforms = WebCL.getPlatforms();
    var platform = platforms[0];
    var devices = platform.getDevices(WebCL.DEVICE_TYPE_ALL);
    var device = devices[0];

    var context = WebCL.createContext({ devices: [device] });

    // 3. Program és kernel
    var program = context.createProgram(kernelSource);
    program.build([device]);
    var kernel = program.createKernel('add_two_numbers');

    // 4. Adatok
    var aArray = new Float32Array([5]);
    var bArray = new Float32Array([10]);
    var cArray = new Float32Array([0]);

    var bufA = context.createBuffer(
      WebCL.MEM_READ_ONLY | WebCL.MEM_COPY_HOST_PTR,
      aArray.byteLength,
      aArray,
    );
    var bufB = context.createBuffer(
      WebCL.MEM_READ_ONLY | WebCL.MEM_COPY_HOST_PTR,
      bArray.byteLength,
      bArray,
    );
    var bufC = context.createBuffer(WebCL.MEM_WRITE_ONLY, cArray.byteLength);

    // 5. Kernel argumentumok
    kernel.setArg(0, bufA);
    kernel.setArg(1, bufB);
    kernel.setArg(2, bufC);

    // 6. Command queue és kernel futtatás
    var queue = context.createCommandQueue(device);
    queue.enqueueNDRangeKernel(kernel, 1, null, [1], null);
    queue.finish();

    // 7. Eredmény visszaolvasása
    queue.enqueueReadBuffer(bufC, true, 0, cArray.byteLength, cArray);

    document.getElementById('result').innerText =
      'Eredmény: ' + aArray[0] + ' + ' + bArray[0] + ' = ' + cArray[0];
  } catch (e) {
    document.getElementById('result').innerText = 'Hiba: ' + e.message;
  }
}

// Futtatás
runWebCL();
