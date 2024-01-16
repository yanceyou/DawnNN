// const input = (await fetch("input.1.3.224.224.bin")
//   .then(response => response.arrayBuffer())
//   .then(arrayBuffer => {
//     // 读取的二进制数据存储在arrayBuffer中

//     // 创建一个DataView对象，用于解析二进制数据
//     var dataView = new DataView(arrayBuffer);

//     // 创建一个浮点数数组，用于存储解析后的数据
//     var intArray = [];


//     // 循环读取二进制数据，每次读取4个字节（32位整数）
//     for (var i = 0; i < arrayBuffer.byteLength; i += 1) {
//       // 使用getInt32方法从DataView中读取整数
//       var intValue = dataView.getInt8(i); // 第二个参数表示是否使用小端字节序

//       // 将读取的整数添加到数组中
//       intArray.push(intValue);
//     }

//     return intArray;
//   })
//   .catch(error => {
//     // 处理错误
//     console.error('Error:', error);
//   }))!;

// async function init() {
//   if (!navigator.gpu) {
//     console.error("Not approve WebGPU")
//     return
//   }
//   const adapter = await navigator.gpu.requestAdapter()
//   if (!adapter) {
//     console.error("Not approve WebGPU")
//     return
//   }
//   const device = await adapter.requestDevice()
//   console.log("Adapter", adapter)
//   console.log("Device", device)
//   return device
// }

// const inputData = new Int32Array(input);

// console.log(inputData.length, inputData.byteLength, inputData);

// const device = (await init())!;
// const buffer = device.createBuffer({
//   label: "input buffer",
//   size: inputData.byteLength,
//   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
// });

// {
//   let start = performance.now();
//   device.queue.writeBuffer(buffer, 0, inputData);
//   console.log(`writeBuffer cost=${performance.now() - start}`)
// }

// const outputBuffer = device.createBuffer({
//   label: "output buffer",
//   size: inputData.length * Float32Array.BYTES_PER_ELEMENT,
//   usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
// });

// const readBuffer = device.createBuffer({
//   label: "read buffer",
//   size: inputData.length * Float32Array.BYTES_PER_ELEMENT,
//   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
// });

// const readBuffer2 = device.createBuffer({
//   label: "read buffer",
//   size: inputData.length * Float32Array.BYTES_PER_ELEMENT,
//   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
// });

// function createSubMeanPass(encoder: GPUCommandEncoder) {
//   const module = device.createShaderModule({
//     label: "submean shader",
//     code: `
//       @group(0) @binding(0) var<storage, read_write> input: array<i32>;
//       @group(0) @binding(1) var<storage, read_write> output: array<f32>;

//       @compute @workgroup_size(32)
//       fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
//         output[global_id.x] = f32(input[global_id.x]) / 255.0f - 0.5f;
//       }
//     `
//   });

//   const pipeline = device.createComputePipeline({
//     label: "submean pipeline",
//     layout: "auto",
//     compute: {
//       module: module,
//       entryPoint: "computeMain",
//     }
//   });

//   const bindGroup = device.createBindGroup({
//     layout: pipeline.getBindGroupLayout(0),
//     entries: [
//       { binding: 0, resource: { buffer: buffer } },
//       { binding: 1, resource: { buffer: outputBuffer } },
//     ],
//   });

//   const computePass = encoder.beginComputePass();
//   computePass.setPipeline(pipeline);
//   computePass.setBindGroup(0, bindGroup);
//   const workgroupCount = Math.ceil(input.length / 32);
//   computePass.dispatchWorkgroups(workgroupCount);
//   computePass.end();
// }

// function createConvPass(encoder: GPUCommandEncoder) {
//   const module = device.createShaderModule({
//     label: "conv2d shader",
//     code: `
//       @group(0) @binding(0) var<storage, read_write> input: array<f32>;

//       @compute @workgroup_size(32)
//       fn computeMain(@builtin(global_invocation_id) global_id: vec3u) {
//         input[global_id.x] = (input[global_id.x] + 0.5f) * 255;
//       }
//     `,
//   })
//   const pipeline = device.createComputePipeline({
//     label: "conv2d pipeline",
//     layout: "auto",
//     compute: {
//       module: module,
//       entryPoint: "computeMain"
//     }
//   });
//   const bindGroup = device.createBindGroup({
//     label: "conv2d bindGroup",
//     layout: pipeline.getBindGroupLayout(0),
//     entries: [
//       { binding: 0, resource: { buffer: outputBuffer } },
//     ]
//   });
//   const pass = encoder.beginComputePass();
//   pass.setPipeline(pipeline);
//   pass.setBindGroup(0, bindGroup);
//   const workgroupCount = Math.ceil(input.length / 32);
//   pass.dispatchWorkgroups(workgroupCount);
//   pass.end();
// }

// export async function main() {
//   {
//     let start = performance.now();
//     const encoder = device.createCommandEncoder();

//     createSubMeanPass(encoder);
//     encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, outputBuffer.size);

//     createConvPass(encoder);
//     encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer2, 0, outputBuffer.size);

//     device.queue.submit([encoder.finish()]);
//     console.log(`submit cost=${performance.now() - start}`)
//   }

//   {
//     let start = performance.now();
//     await readBuffer.mapAsync(GPUMapMode.READ);
//     let output = new Float32Array(readBuffer.getMappedRange());
//     console.log(`mapAsync output cost=${performance.now() - start}`)
//     console.log(output.length, output.byteLength, output);
//     // for (let i = 0; i < output.length; i++) {
//     //   const target = (input[i] / 255.0) - 0.5;
//     //   if (Math.abs(target - output[i]) > 0.0000001) {
//     //     console.log(i, target, output[i])
//     //   }
//     // }
//     console.log("output end")
//   }

//   {
//     let start = performance.now();
//     await readBuffer2.mapAsync(GPUMapMode.READ);
//     let output = new Float32Array(readBuffer2.getMappedRange());
//     console.log(`mapAsync output cost=${performance.now() - start}`)
//     console.log(output.length, output.byteLength, output);
//     // for (let i = 0; i < output.length; i++) {
//     //   const target = (input[i] / 255.0) - 0.5;
//     //   if (Math.abs(target - output[i]) > 0.0000001) {
//     //     console.log(i, target, output[i])
//     //   }
//     // }
//     console.log("output2 end")
//   }

//   console.log("end")
// }

// main()

// Import @tensorflow/tfjs or @tensorflow/tfjs-core
import * as tf from '@tensorflow/tfjs';
// Add the WebGPU backend to the global backend registry.
import '@tensorflow/tfjs-backend-webgpu';
// Set the backend to WebGPU and wait for the module to be ready.

const data = (await fetch("input.1.3.224.224.bin")
  .then(response => response.arrayBuffer())
  .then(arrayBuffer => {
    // 读取的二进制数据存储在arrayBuffer中

    // 创建一个DataView对象，用于解析二进制数据
    var dataView = new DataView(arrayBuffer);

    // 创建一个浮点数数组，用于存储解析后的数据
    var intArray = [];


    // 循环读取二进制数据，每次读取4个字节（32位整数）
    for (var i = 0; i < arrayBuffer.byteLength; i += 1) {
      // 使用getInt32方法从DataView中读取整数
      var intValue = dataView.getInt8(i); // 第二个参数表示是否使用小端字节序

      // 将读取的整数添加到数组中
      intArray.push(intValue);
    }

    return intArray;
  })
  .catch(error => {
    // 处理错误
    console.error('Error:', error);
  }))!;

const conv_0_b = (await fetch("conv_0_b.json")
  .then(response => response.json())
  .then(data => {
    // 在这里可以使用解析后的 JSON 数据
    console.log(data);
    return data;
  })
  .catch(error => {
    // 处理错误
    console.error('Error:', error);
  }))!;

const conv_0_w = (await fetch("conv_0_w.json")
  .then(response => response.json())
  .then(data => {
    // 在这里可以使用解析后的 JSON 数据
    console.log(data);
    return data;
  })
  .catch(error => {
    // 处理错误
    console.error('Error:', error);
  }))!;

tf.setBackend('webgpu').then(() => {
  // const input = tf.tensor4d(data, [1, 224, 224, 3]);
  // // const mean = tf.mean(input, axis); // 根据需要指定均值计算的轴
  // 创建一个形状为 [1, 3, 3, 1] 的输入张量。
  const input = tf.tensor4d(data, [1, 224, 224, 3]);
  const inputNHWC = tf.sub(tf.div(input, 255.0), 0.5).as4D(1, 224, 224, 3);
  inputNHWC.print();
  console.log(inputNHWC.shape);

  const inputNCHW = tf.transpose(inputNHWC, [0, 3, 1, 2]);
  inputNCHW.print();

  // 创建一个形状为 [2, 2, 1, 1] 的过滤器张量。
  const filterNCHW = tf.tensor4d(conv_0_w, [56, 3, 3, 3]);
  filterNCHW.print();
  console.log(filterNCHW.shape);

  const filterNHWC = tf.transpose(filterNCHW, [2, 3, 1, 0]);
  filterNHWC.print();
  console.log(filterNHWC.shape);

  const bias = tf.tensor1d(conv_0_b);
  bias.print();

  // 执行卷积操作。
  const resultNHWC = tf.conv2d(inputNHWC, filterNHWC, 2, 1, "NHWC");
  resultNHWC.print();
  const outputNHWC = resultNHWC.add(bias);
  outputNHWC.print();
  console.log("NHWC", inputNHWC.shape, filterNHWC.shape, resultNHWC.shape, outputNHWC.shape);

  const finalNCHW = outputNHWC.transpose([0, 3, 1, 2]);
  finalNCHW.print();
  console.log(finalNCHW.gather([0]));

  // console.log(inputNCHW.shape, filterNCHW.shape);
  // const resultNCHW = tf.conv2d(inputNCHW, filterNHWC, 2, 1, "NCHW");
  // resultNCHW.print();

  // console.log("NCHW", inputNCHW.shape, filterNCHW.shape, resultNCHW.shape);


  // result.transpose()

  // 添加偏置

  // const filter = tf.tensor(conv_0_w);
  // filter.print();

  // const result = tf.conv2d(normalizedInput, filter, 2, 1, "NHWC", 1);
  // result.print();

  // const weights = [
  //   tf.tensor(conv_0_w),
  //   tf.tensor(conv_0_b)
  // ];
  // const layer = tf.layers.conv2d({
  //   filters: 56,
  //   kernelSize: 3,
  //   inputShape: [1, 3, 224, 224],
  //   strides: 2,
  //   activation: 'relu',
  // });
  // layer.setWeights(weights);
  // console.log(layer.getWeights());
});

