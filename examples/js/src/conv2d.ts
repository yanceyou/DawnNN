
/**
 * A ShaderHelper is a helper class for generating WGSL code.
 */
export interface ShaderHelper {
    /**
     * A helper function to generate the start of main function in WGSL source code.
     *
     * @example
     * const getShaderSource = (shaderHelper: ShaderHelper) => `
     *  ...
     *
     *  ${shaderHelper.mainStart()}
     *    // your code here inside main() function
     *    ...
     *  }
     * `;
     *
     * @param workgroupSize - an optional workgroup size. default is WORKGROUP_SIZE.
     */
    mainStart(workgroupSize?: number | [number, number, number]): string;

    /**
     * A helper function to generate the code snippet for guarding against out-of-bounds size.
     *
     * @example
     * const getShaderSource = (shaderHelper: ShaderHelper) => `
     *  ...
     *
     *  ${shaderHelper.mainStart()}
     *    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
     *
     *    // your code here inside main() function
     *    ...
     *  }
     * `;
     *
     * @param size - the size of the data to guard against. can be a number or a string (WGSL `u32` expression).
     */
    guardAgainstOutOfBoundsWorkgroupSizes(size: unknown): string;

    /**
     * A helper function to generate the code snippet for declaring multiple inputs or outputs.
     *
     * @param variables - an array of IndicesHelper for the variables.
     */
    declareVariables(...variables: IndicesHelper[]): string;

    /**
     * A helper function to register one uniform. Can be called multiple times to register multiple uniforms.
     *
     * @param name - the name of the uniform.
     * @param type - the type of the uniform.
     * @param length - the length of the uniform, default to 1 when it is not provided.
     */
    registerUniform(name: string, type: string, length?: number): ShaderHelper;

    /**
     * A helper function to register multiple uniforms. Can be called multiple times to register multiple uniforms.
     *
     * @param uniforms - an array of uniforms. Each element of the array is an object with 2 properties: `name` and
     *     `type`.
     */
    registerUniforms(uniforms: UniformsArrayType): ShaderHelper;

    /**
     * A helper function to register multiple internal variables. Can be called multiple times to register multiple
     * internal variables.
     *
     * @param variables - an array of IndicesHelper for the variables.
     */
    registerInternalVariables(...variables: IndicesHelper[]): ShaderHelper;
}


/**
 * A set of data that represent a shader program
 */
export interface ProgramInfo {
    /**
     * the name of the program. used for debugging and profiling
     */
    name: string;

    /**
     * an optional object describing the cache information of the program shader.
     *
     * If this is not specified, assume hint is empty and inputDependencies are ['dims'] for all inputs.
     */
    shaderCache?: ProgramShaderCacheInfo;

    /**
     * the shader's processing source code.
     *
     * This function will be called when shader cache missed.
     */
    getShaderSource: (shaderHelper: ShaderHelper) => string;

    /**
     * A function to get run data required to run the program.
     *
     * This function will be called every time the program is executed. Should keep this function as simple as possible.
     */
    getRunData: (inputs: readonly TensorView[]) => {
        outputs: readonly TensorInfo[];
        dispatchGroup: { x: number; y?: number; z?: number };
        programUniforms?: readonly ProgramUniform[];
    };
}

export interface ProgramUniform {
    type: 'int32' | 'float32' | 'uint32';
    data: number | readonly number[];
}

/**
 * a TensorView does not own the data.
 */
export interface TensorView {
    readonly data: number;
    readonly dataType: number;
    readonly dims: readonly number[];

    /**
     * get a Float32Array data view of the tensor data. tensor data must be on CPU.
     */
    getFloat32Array(): Float32Array;

    /**
     * get a BigInt64Array data view of the tensor data. tensor data must be on CPU.
     */
    getBigInt64Array(): BigInt64Array;

    /**
     * get a Int32Array data view of the tensor data. tensor data must be on CPU.
     */
    getInt32Array(): Int32Array;

    /**
     * create a new tensor view with the same data but different dimensions.
     */
    reshape(newDims: readonly number[]): TensorView;
}

export interface InternalActivationAttributes {
    readonly activation: string;
    readonly clipMin?: number;
    readonly clipMax?: number;
    readonly activationCacheKey: string;
}

export interface ConvAttributes extends InternalActivationAttributes, AttributeWithCacheKey {
    readonly autoPad: string;
    readonly dilations: readonly number[];
    readonly format: 'NHWC' | 'NCHW';
    readonly group: number;
    readonly kernelShape: readonly number[];
    readonly pads: readonly number[];
    readonly strides: readonly number[];
    readonly wIsConst: boolean;
}

/**
 * Create a IndicesHelper for an input.
 *
 * @param name - the name of the input.
 * @param type - the tensor type of the input.
 * @param shapeOrRank - the tensor shape or the rank of the input.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the input.
 */
export const inputVariable =
    (name: string, type: number, shapeOrRank: number | readonly number[], components: 1 | 2 | 3 | 4 = 1): IndicesHelper =>
        createIndicesHelper(name, type, shapeOrRank, 'input', components);

const getWgslMappedType = (type: number, components: 1 | 2 | 3 | 4): string | [string, string] => {
    if (components === 3) {
        throw new Error('vec3 has same alignment as vec4, use vec4 instead');
    }

    // return type is [ storage type, runtime type ] or a single string for both
    switch (type) {
        case DataType.float16:
            return components > 1 ? `vec${components}<f16>` : 'f16';
        case DataType.float:
            return components > 1 ? `vec${components}<f32>` : 'f32';
        case DataType.int32:
            return components > 1 ? `vec${components}<i32>` : 'i32';
        case DataType.uint32:
            return components > 1 ? `vec${components}<u32>` : 'u32';
        case DataType.int64:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'i32'];
        case DataType.uint64:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'u32'];
        case DataType.bool:
            if (components !== 4) {
                throw new Error('bool must be vec4');
            }
            return ['u32', 'vec4<bool>'];

        default:
            throw new Error(`Unknown data type: ${type}`);
    }
};

export const tensorTypeToWsglStorageType = (type: DataType, components: 1 | 2 | 3 | 4 = 1) => {
    const mappedType = getWgslMappedType(type, components);
    return typeof mappedType === 'string' ? mappedType : mappedType[0];
};

export const typeSnippet = (component: number, dataType: string) => {
    switch (component) {
        case 1:
            return dataType;
        case 2:
            return `vec2<${dataType}>`;
        case 3:
            return `vec3<${dataType}>`;
        case 4:
            return `vec4<${dataType}>`;
        default:
            throw new Error(`${component}-component is not supported.`);
    }
};

export const biasSnippet = (hasBias: boolean): string => `
        ${hasBias ? 'value = value + getBiasByOutputCoords(coords);' : ''}
        `;

const conv2dCommonSnippet =
    (isChannelsLast: boolean, fitAOuter: boolean, fitBOuter: boolean, fitInner: boolean, addBias = false,
        attributes: ConvAttributes, innerElementSizeX = 4, innerElementSizeW = 4, innerElementSize = 4,
        dataType = 'f32'): string => {
        const getXSnippet = (innerElementSize: number) => {
            switch (innerElementSize) {
                case 1:
                    return 'resData = x[xIndex];';
                case 3:
                    return `resData = vec3<${dataType}>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);`;
                case 4:
                    return 'resData = x[xIndex / 4];';
                default:
                    throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
            }
        };
        const getWSnippet = (innerElementSize: number) => {
            switch (innerElementSize) {
                case 1:
                    return 'return w[row * i32(uniforms.w_shape[3]) + colIn];';
                case 4:
                    return 'return w[row * i32(uniforms.w_shape[3]) / 4 + colIn];';
                default:
                    throw new Error(`innerElementSize ${innerElementSize} is not supported.`);
            }
        };
        const coordASnippet = isChannelsLast ? `
    let coord = vec4<i32>(batch, xRow, xCol, xCh);
    ` :
            `
    let coord = vec4<i32>(batch, xCh, xRow, xCol);
    `;

        const coordResSnippet = isChannelsLast ? `
    let coords = vec4<i32>(
      batch,
      row / outWidth,
      row % outWidth,
      col);
    ` :
            `
    let coords = vec4<i32>(
      batch,
      row,
      col / outWidth,
      col % outWidth);
    `;

        const xHeight = isChannelsLast ? 'i32(uniforms.x_shape[1])' : 'i32(uniforms.x_shape[2])';
        const xWidth = isChannelsLast ? 'i32(uniforms.x_shape[2])' : 'i32(uniforms.x_shape[3])';
        const row = isChannelsLast ? 'row' : 'col';
        const col = isChannelsLast ? 'col' : 'row';
        const readXSnippet = `
    let inChannels = i32(uniforms.w_shape[2]);
    let outWidth = ${isChannelsLast ? 'i32(uniforms.result_shape[2])' : 'i32(uniforms.result_shape[3])'};
    let outRow = ${row} / outWidth;
    let outCol = ${row} % outWidth;

    let WRow = ${col} / (filterDims[1] * inChannels);
    let WCol = ${col} / inChannels % filterDims[1];
    let xRow = outRow * stride[0] + dilation[0] * WRow - pad[0];
    let xCol = outCol * stride[1] + dilation[1] * WCol - pad[1];
    let xCh = ${col} % inChannels;
    var resData = ${typeSnippet(innerElementSizeX, dataType)}(0.0);
    // The bounds checking is always needed since we use it to pad zero for
    // the 'same' padding type.
    if (xRow >= 0 && xRow < ${xHeight} && xCol >= 0 && xCol < ${xWidth}) {
      ${coordASnippet}
      let xIndex = getIndexFromCoords4D(coord, vec4<i32>(uniforms.x_shape));
      ${getXSnippet(innerElementSizeX)}
    }
    return resData;`;
        const sampleX = isChannelsLast
            ? (fitAOuter && fitInner ? `
                let col = colIn * ${innerElementSizeX};
                ${readXSnippet}`
                : ` let col = colIn * ${innerElementSizeX};
                if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
                    ${readXSnippet}
                }
                return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`)
            : (fitInner && fitBOuter ? `
                let col = colIn * ${innerElementSizeX};
                ${readXSnippet}` :
                `
                let col = colIn * ${innerElementSizeX};
                if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
                ${readXSnippet}
                }
                return ${typeSnippet(innerElementSizeX, dataType)}(0.0);`
            );

        const sampleW = `${getWSnippet(innerElementSizeW)}`;

        const resType = typeSnippet(innerElementSize, dataType);
        const aType =
            isChannelsLast ? typeSnippet(innerElementSizeX, dataType) : typeSnippet(innerElementSizeW, dataType);
        const bType =
            isChannelsLast ? typeSnippet(innerElementSizeW, dataType) : typeSnippet(innerElementSizeX, dataType);

        return `
fn mm_readA(batch: i32, row : i32, colIn : i32) -> ${aType} {
    ${isChannelsLast ? sampleX : sampleW}
}

fn mm_readB(batch: i32, row : i32, colIn : i32) -> ${bType} {
    ${isChannelsLast ? sampleW : sampleX}
}

fn mm_write(batch: i32, row : i32, colIn : i32, valueIn : ${resType}) {
    let col = colIn * ${innerElementSize};
    if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
        var value = valueIn;
        let outWidth = ${isChannelsLast ? 'i32(uniforms.result_shape[2])' : 'i32(uniforms.result_shape[3])'};
        ${coordResSnippet}
        ${biasSnippet(addBias)}
        value = max(value, ${resType}(0.0));
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
    }
}`;
    };

export const createConv2DMatMulProgramInfo =
    (inputs: readonly TensorView[], attributes: ConvAttributes, outputShape: readonly number[], dimAOuter: number,
        dimBOuter: number, dimInner: number, hasBias: boolean, sequentialAccessByThreads: boolean): ProgramInfo => {

        const isChannelsLast = attributes.format === 'NHWC';
        const inChannels = isChannelsLast ? inputs[0].dims[3] : inputs[0].dims[1];
        const batchSize = outputShape[0];
        const outWidth = isChannelsLast ? outputShape[2] : outputShape[3];
        const outHeight = isChannelsLast ? outputShape[1] : outputShape[2];
        const outChannels = isChannelsLast ? outputShape[3] : outputShape[1];

        // TODO: enable vec4 for NCHW
        const isVec4 = isChannelsLast && (inChannels % 4 === 0 || inChannels % 3 === 0) && outChannels % 4 === 0;

        // TODO: fine tune size
        const dispatchX = isChannelsLast ? outChannels : outWidth * outHeight;
        const dispatchY = isChannelsLast ? outWidth * outHeight : outChannels;
        const workGroupSize: [number, number, number] = [8, 8, 1];
        const elementsPerThread = dimAOuter <= 8 ? [4, 1, 1] : [4, 4, 1];
        const dispatch = [
            Math.ceil(dispatchX / workGroupSize[0] / elementsPerThread[0]),
            Math.ceil(dispatchY / workGroupSize[1] / elementsPerThread[1]),
            Math.ceil(batchSize / workGroupSize[2] / elementsPerThread[2])
        ];

        // LOG_DEBUG('verbose', () => `[conv2d_mm_webgpu] dispatch = ${dispatch}`);

        const innerElementSize = isVec4 ? (isChannelsLast && inChannels % 4 !== 0 ? 3 : 4) : 1;

        const tileAOuter = workGroupSize[1] * elementsPerThread[1];
        const tileBOuter = workGroupSize[0] * elementsPerThread[0];
        const tileInner = Math.max(workGroupSize[0] * innerElementSize, workGroupSize[1]);

        const fitAOuter = dimAOuter % tileAOuter === 0;
        const fitBOuter = dimBOuter % tileBOuter === 0;
        const fitInner = dimInner % tileInner === 0;

        const elementsSize = isVec4 ? [innerElementSize, 4, 4] : [1, 1, 1];
        const t = tensorTypeToWsglStorageType(inputs[0].dataType);

        // TODO: support component 2, 3.
        const components = isVec4 ? 4 : 1;
        const programUniforms: ProgramUniform[] = [
            { type: 'int32', data: dimAOuter },
            { type: 'int32', data: dimBOuter },
            { type: 'int32', data: dimInner }
        ];

        const inputVariables = [
            inputVariable('x', inputs[0].dataType, inputs[0].dims.length, innerElementSize === 3 ? 1 : innerElementSize),
            inputVariable('w', inputs[1].dataType, inputs[1].dims.length, components),
            inputVariable('bias', inputs[2].dataType, inputs[2].dims.length, components)
        ];

        programUniforms.push(...createTensorShapeVariables(inputs[0].dims));
        programUniforms.push(...createTensorShapeVariables(inputs[1].dims));
        programUniforms.push(...createTensorShapeVariables(inputs[2].dims));

        const output = outputVariable('result', inputs[0].dataType, outputShape.length, components);
        programUniforms.push(...createTensorShapeVariables(outputShape));

        return {
            name: 'Conv2DMatMul',
            shaderCache: { hint: attributes.cacheKey },
            getRunData: () => ({
                outputs: [{ dims: outputShape, dataType: inputs[0].dataType }],
                dispatchGroup: { x: dispatch[0], y: dispatch[1], z: dispatch[2] },
                programUniforms,
            }),
            getShaderSource: (shaderHelper: ShaderHelper) => `
        fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
            return dot(coords, vec4<i32>(shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
        }
        fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
            return dot(coords, vec4<i32>(i32(uniforms.result_strides.x), i32(uniforms.result_strides.y), i32(uniforms.result_strides.z), 1));
        }
        // struct Uniforms { xShape : vec4<i32>, wShape : vec4<i32>, outShape : vec4<i32>,
        // outShapeStrides: vec3<i32>, filterDims : vec2<i32>, pad : vec2<i32>, stride : vec2<i32>,
        // dilation : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32 };
        ${shaderHelper.registerUniform('dimAOuter', 'i32')
                    .registerUniform('dimBOuter', 'i32')
                    .registerUniform('dimInner', 'i32')
                    .declareVariables(...inputVariables, output)}

        const filterDims : vec2<i32> = vec2<i32>(${attributes.kernelShape[0]}, ${attributes.kernelShape[1]});
        const pad : vec2<i32> = vec2<i32>(${attributes.pads[0]}, ${attributes.pads[1]});
        const stride : vec2<i32> = vec2<i32>(${attributes.strides[0]}, ${attributes.strides[1]});
        const dilation : vec2<i32> = vec2<i32>(${attributes.dilations[0]}, ${attributes.dilations[1]});

        fn setOutputAtIndex(flatIndex : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
            result[flatIndex] = ${isVec4 ? `vec4<${t}>` : t}(value);
        }
        fn setOutputAtCoords(d0 : i32, d1 : i32, d2 : i32, d3 : i32, value : ${isVec4 ? `vec4<${t}>` : t}) {
            let flatIndex = getOutputIndexFromCoords(vec4<i32>(d0, d1, d2, d3));
            setOutputAtIndex(flatIndex ${isVec4 ? '/ 4' : ''}, value);
        }
        fn getBiasByOutputCoords(coords : vec4<i32>) -> ${isVec4 ? `vec4<${t}>` : t} {
            return bias[coords.${isChannelsLast ? 'w' : 'y'}${isVec4 ? '/ 4' : ''}];
        }

        ${conv2dCommonSnippet(isChannelsLast, fitAOuter, fitBOuter, fitInner, hasBias, attributes, elementsSize[0], elementsSize[1], elementsSize[2], t)}
        ${isVec4 ? makeMatMulPackedVec4Source(elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner)
                    : makeMatMulPackedSource(
                        elementsPerThread, workGroupSize, t, undefined, !isChannelsLast, tileInner, false, undefined,
                        sequentialAccessByThreads)}
`
        };
    };
