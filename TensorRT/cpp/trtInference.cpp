#include"trtInference.hpp"

TensorrtInference::TensorrtInference(){
    
}

TensorrtInference::~TensorrtInference(){
    // TensorrtInference* infer = new TensorrtInference();
    // delete infer;
    runtime->destroy();
    engine->destroy();
    context->destroy();
    cudaStreamSynchronize(stream);
    // 释放流和缓存
    cudaStreamDestroy(stream);
    for(int i = 0;i<4; ++i){
        CHECK(cudaFree(buffers[i]));
    }
}
float* TensorrtInference::blobFromImage(cv::Mat& img){
    float* blob = new float[img.total()*3];
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (float)img.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
    return blob;
}


int TensorrtInference::initModel(string engine_path,int batch){
    batch_size = batch;
    // 构建模型
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::ifstream file(engine_path, std::ios::binary);
    if (file.good()) {  // 判断file文件是否可读取
        file.seekg(0, file.end);    // 将文件指针设置为文件的尾部，获取size
        size = file.tellg(); 
        file.seekg(0, file.beg);    // 将文件指针设置为文件的开头读取数据
        trtModelStream = new char[size];    // 分配大小为size的缓冲区
        assert(trtModelStream); 
        file.read(trtModelStream, size);    // 将文件的内容读取到缓冲区中
        file.close();   // 关闭文件
    }
    runtime = createInferRuntime(logger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();  // 创建执行上下文
    assert(context != nullptr);

}
int TensorrtInference::onnxToEngine(const char* onnx_path,string save_engine_path,int min_batch,int medium_batch,int max_batch){
    // 创建builder
    IBuilder* builder = nvinfer1::createInferBuilder(logger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // 创建网络，是否以显式批次的方式定义网络。
    INetworkDefinition* network = builder->createNetworkV2(EXPLICIT_BATCH);

    /*
        解析onnx
        nvonnxparser::createParser工厂方法创建一个IParser对象，并将network和m_logger作为参数进行初始化。
        使用parser->parseFromFile从本地文件夹中解析出ONX模型，同时指定ILogger的严重程度或者警告级别，以确保不会显示不必要的日志。
    */
    IParser* parser = nvonnxparser::createParser(*network, logger);
    bool parser_status = parser->parseFromFile(onnx_path, static_cast<int>(ILogger::Severity::kWARNING));

    // 获取网络的输入
    IOptimizationProfile* profile = builder->createOptimizationProfile();   // 创建profile设置输入的大小和批次
    for(int i=0;i<input_names.size();++i){
        Dims dim = network->getInput(i)->getDimensions();  
        if (dim.d[0] == -1)  // -1代表是动态输入
        {
            const char* name = network->getInput(i)->getName(); // 获取网络输入的名称
            profile->setDimensions(name, OptProfileSelector::kMIN, Dims4(min_batch, dim.d[1], dim.d[2], dim.d[3])); // 最小输入
            profile->setDimensions(name, OptProfileSelector::kOPT, Dims4(medium_batch, dim.d[1], dim.d[2], dim.d[3])); // 建议输入
            profile->setDimensions(name, OptProfileSelector::kMAX, Dims4(max_batch, dim.d[1], dim.d[2], dim.d[3])); // 最大输入
            config->addOptimizationProfile(profile);
        }
    }
    
    // 构建 engine
    config->setMaxWorkspaceSize(1 << 20); // 设置工作空间大小
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    // 序列化模型到 engine file
    IHostMemory* modelStream{ nullptr };
    assert(engine != nullptr);
    modelStream = engine->serialize();

    ofstream p(save_engine_path, std::ios::binary);
    if (!p) {
        cerr << "could not open output file to save model" << endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size()); // 将modelStream指向的内存中的数据写入到p中
    cout << "generate file success!" << endl;

    // 释放内存
    modelStream->destroy();
    network->destroy();
    engine->destroy();
    builder->destroy();
    config->destroy();
    return 0;
}

int TensorrtInference::doInference(vector<float*> input,vector<float*> &output){

    // const ICudaEngine& engine_infer = context->getEngine();    // 获取推理引擎
    // 指向要传递给引擎的输入和输出设备缓冲区的指针。
    // 引擎需要的缓冲区数量恰好是 IEngine::getNbBindings()个。
    // cout<<engine->getNbBindings()<<endl;
    // assert(engine.getNbBindings() == 4);

    // 创建流
    CHECK(cudaStreamCreate(&stream));

    // 为了绑定缓冲区，我们需要知道输入和输出张量的名称。
    for(auto in_:input_names){
        int inputIndex = engine->getBindingIndex(in_.first.c_str());
        Dims dims = context->getBindingDimensions(inputIndex);
        dims.d[0]=batch_size;
        // 请注意，索引保证小于 IEngine::getNbBindings()
        context->setBindingDimensions(inputIndex, dims);
        // 创建GPU缓冲区
        CHECK(cudaMalloc(&buffers[inputIndex], batch_size * 3 * in_.second[0] * in_.second[1] * sizeof(float)));

        // DMA 将批处理数据输入到设备，异步推断批处理，DMA 输出回主机
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input[inputIndex], batch_size * 3 * in_.second[0] * in_.second[1] * sizeof(float), cudaMemcpyHostToDevice, stream));

    }

    for(auto out_:output_names){
        int outputIndex = engine->getBindingIndex(out_.c_str());
        Dims dims = context->getBindingDimensions(outputIndex);
        CHECK(cudaMalloc(&buffers[outputIndex], batch_size * dims.d[1] * sizeof(float)));
    }
    context->enqueue(batch_size, buffers, stream, nullptr); // 排队执行批量大小

    int out_idx=0;
    for(auto out_:output_names){
        int outputIndex = engine->getBindingIndex(out_.c_str());
        Dims dims = context->getBindingDimensions(outputIndex);
        CHECK(cudaMemcpyAsync(output[out_idx], buffers[outputIndex], batch_size * dims.d[1] * sizeof(float), cudaMemcpyDeviceToHost, stream));
        out_idx+=1;
    }
    cout<<output[0][0]<<" "<<output[0][1]<<" "<<output[1][0]<<" "<<output[1][1]<<endl;
    
    
    return 0;
}