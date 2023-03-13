#include "tensorrt.hpp"

TensorRT::TensorRT(){

}

TensorRT::~TensorRT(){

}

inline int64_t volume(const Dims& d) {
    if (d.nbDims == 0) {
        return 0;
    }
    bool skip_first = d.d[0] <= 0;
    if (skip_first == true && d.nbDims == 1) {
        return 0;
    }
    int64_t v = 1;
    int start_index = skip_first ? 1 : 0;
    for (int64_t i = start_index; i < d.nbDims; i++) v *= d.d[i];
    return v;
}

inline unsigned int elementSize(DataType t) {
    switch (t) {
        case DataType::kINT32:
            // Fallthrough, same as kFLOAT
        case DataType::kFLOAT:
            return 4;
        case DataType::kHALF:
            return 2;
        case DataType::kINT8:
            return 1;
    }
    assert(0);
    return 0;
}
int TensorRT::loadModel(string modelPath,int batchSize){
    int code = -1;
    if(modelPath.find("onnx")!=std::string::npos){
        code = loadOnnxModel(modelPath);
    }
    else if(modelPath.find("trt")!=std::string::npos){
        code = loadTrtModel(modelPath,batchSize);
    }
    for(auto inputName:inputNames){
        const int inputIndex = engine_->getBindingIndex(inputName.c_str());
        Dims dims1 = context_->getBindingDimensions(inputIndex);
        DataType data_type1 = engine_->getBindingDataType(inputIndex);
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers_[inputIndex], volume(dims1) * elementSize(data_type1)));
    }
    for(auto outputName:outputNames){
        const int outputIndex = engine_->getBindingIndex(outputName.c_str());
        Dims dims2 = context_->getBindingDimensions(outputIndex);
        DataType data_type2 = engine_->getBindingDataType(outputIndex);
        // Create GPU buffers on device
        CHECK(cudaMalloc(&buffers_[outputIndex], volume(dims2) * elementSize(data_type2)));
    }
    // Create stream
    CHECK(cudaStreamCreate(&stream_));

    return code;
}

int TensorRT::doInference(float* input, const int batch_size, std::vector<void*>& outputs) {

    // collect output size info
    for (int i = 0; i < inputNames.size(); i++) {
        int inputIndex = engine_->getBindingIndex(inputNames[i].c_str());
        Dims inputDims = context_->getBindingDimensions(inputIndex);
        inputIndexes_.push_back(inputIndex);
        inputDims_.push_back(inputDims);
        inputSizes_.push_back(volume(inputDims));
    }

    for (size_t i = 0; i < outputNames.size(); i++) {
        int outputIndex = engine_->getBindingIndex(outputNames[i].c_str());
        Dims outputDims = context_->getBindingDimensions(outputIndex);
        outputIndexes_.push_back(outputIndex);
        outputDims_.push_back(outputDims);
        outputSizes_.push_back(volume(outputDims));
        outputPtrs_.push_back(malloc(volume(outputDims) * sizeof(float)));
    }
    // void *buffers[2];
    for(int i=0;i<inputNames.size();++i){
        CHECK(cudaMalloc(&buffers_[inputIndexes_[i]], inputSizes_[i] * sizeof(float)));
        CHECK(cudaMemcpyAsync(buffers_[inputIndexes_[i]], input, inputSizes_[i] * sizeof(float), cudaMemcpyHostToDevice, stream_));
    }

    for(int i=0;i<outputNames.size();++i){
        CHECK(cudaMalloc(&buffers_[outputIndexes_[i]], outputSizes_[i] * sizeof(float)));
    }
     // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    context_->enqueue(batch_size, buffers_, stream_, nullptr);
    for(int i=0;i<outputNames.size();++i){
        CHECK(cudaMemcpyAsync(outputPtrs_[i], buffers_[outputIndexes_[i]], outputSizes_[i] * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }
    
    cudaStreamSynchronize(stream_);

    // Release stream and buffers
    cudaStreamDestroy(stream_);
    CHECK(cudaFree(buffers_[0]));
    CHECK(cudaFree(buffers_[1]));
    float *c = (float *)outputPtrs_[0];
    outputs = outputPtrs_;
    std::cout<<c[0]<<" "<<c[1]<<std::endl;

    std::cout<<123<<std::endl;
    return 0;
}


int TensorRT::loadOnnxModel(string modelPath){

    // 创建builder
    IBuilder* builder = nvinfer1::createInferBuilder(logger_);
    IBuilderConfig* config = builder->createBuilderConfig();
    // 创建网络，是否以显式批次的方式定义网络。
    static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);
    /*
        解析onnx
        nvonnxparser::createParser工厂方法创建一个IParser对象，并将network和m_logger作为参数进行初始化。
        使用parser->parseFromFile从本地文件夹中解析出ONX模型，同时指定ILogger的严重程度或者警告级别，以确保不会显示不必要的日志。
    */
    IParser* parser = nvonnxparser::createParser(*network, logger_);
    bool parser_status = parser->parseFromFile(modelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING));
    // 获取网络的输入
    IOptimizationProfile* profile = builder->createOptimizationProfile();   // 创建profile设置输入的大小和批次
    for(int i=0;i<inputNames.size();++i){
        Dims dim = network->getInput(i)->getDimensions();
        if (dim.d[0] == -1)  // -1代表是动态输入
        {
            const char* name = network->getInput(i)->getName(); // 获取网络输入的名称
            profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(minBatch, dim.d[1], dim.d[2], dim.d[3])); // 最小输入
            profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(mediumBatch, dim.d[1], dim.d[2], dim.d[3])); // 建议输入
            profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(maxBatch, dim.d[1], dim.d[2], dim.d[3])); // 最大输入
        }
    }
    
    config->addOptimizationProfile(profile);
    // 构建 engine
    config->setMaxWorkspaceSize(1 << 20); // 设置工作空间大小
    engine_ = builder->buildEngineWithConfig(*network, *config);
    
    // 序列化模型到 engine file
    IHostMemory *modelStream{ nullptr };
    assert(engine_ != nullptr);

    modelStream = engine_->serialize();
    context_ = engine_->createExecutionContext();
    int index = modelPath.find(".onnx");
    ofstream p(modelPath.replace(index,5,".trt"), std::ios::binary);
    if (!p) {
        cerr << "could not open output file to save model" << endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); // 将modelStream指向的内存中的数据写入到p中
    cout << "generate file success!" << endl;

    // 释放内存
    modelStream->destroy();
    network->destroy();
    builder->destroy();
    config->destroy();
    parser->destroy();
    return 0;
}


int TensorRT::loadTrtModel(string modelPath,int batchSize){
    // 构建模型
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(modelPath, std::ios::binary);
    if (file.good()) {  // 判断file文件是否可读取
        file.seekg(0, file.end);    // 将文件指针设置为文件的尾部，获取size
        size = file.tellg(); 
        file.seekg(0, file.beg);    // 将文件指针设置为文件的开头读取数据
        trtModelStream = new char[size];    // 分配大小为size的缓冲区
        assert(trtModelStream); 
        file.read(trtModelStream, size);    // 将文件的内容读取到缓冲区中
        file.close();   // 关闭文件
    }
    IRuntime *runtime = createInferRuntime(logger_);
    assert(runtime != nullptr);
    engine_ = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine_ != nullptr);
    context_ = engine_->createExecutionContext();  // 创建执行上下文
    // int input_index = engine->getBindingIndex(inputName.c_str());
    // Dims dims = context->getBindingDimensions(input_index);
    // std::cout << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;
    // dims.d[0] = 4;
    // CHECK(context->setBindingDimensions(input_index, dims));
    std::cout<<"successfully"<<std::endl;
    assert(context_ != nullptr);
    runtime->destroy();
    return 0;
}


