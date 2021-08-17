#include <assert.h>

#include <chrono>
#include <vector>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

void simple_net(engine::kind engine_kind, int times = 100) {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    engine eng(engine_kind, 0);
    stream s(eng);

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    //[Create network]

    const memory::dim batch = 1;
    const memory::dim ic = 48;
    const memory::dim ih = 480;
    const memory::dim iw = 270;
    const memory::dim oc = 64;
    const memory::dim oh = 478;
    const memory::dim ow = 268;
    const memory::dim kh = 3;
    const memory::dim kw = 3;
    const memory::dim stride = 1;
    const memory::dim padding = 0;

    memory::dims conv1_src_tz = {batch, ic, iw, ih};
    memory::dims conv1_weights_tz = {oc, ic, kw, kh};
    memory::dims conv1_bias_tz = {oc};
    memory::dims conv1_dst_tz = {batch, oc, ow, oh};
    memory::dims conv1_strides = {stride, stride};
    memory::dims conv1_padding = {padding, padding};

    std::vector<float> user_src(batch * ic * iw * ih);
    std::vector<float> user_dst(batch * 1000);
    std::vector<float> conv1_weights(product(conv1_weights_tz));
    std::vector<float> conv1_bias(product(conv1_bias_tz));

    auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
    write_to_dnnl_memory(user_src.data(), user_src_memory);
    auto user_weights_memory = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
    write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
    auto conv1_user_bias_memory = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);

    auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);
    //[Create convolution memory descriptors]

   auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_winograd, conv1_src_md, conv1_weights_md,
            conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
            conv1_padding);
    //[Create convolution descriptor]

   auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    //[Create convolution primitive descriptor]

   auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_memory},
                {DNNL_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory)
                .execute(s, user_weights_memory, conv1_weights_memory);
    }
    //[Reorder data and weights]

   auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    //[Create memory for output]

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv1_src_memory},
            {DNNL_ARG_WEIGHTS, conv1_weights_memory},
            {DNNL_ARG_BIAS, conv1_user_bias_memory},
            {DNNL_ARG_DST, conv1_dst_memory}});

    ///warm up
    for (int i = 0; i < 1; ++i) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }


    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    for (int j = 0; j < times; ++j) {
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
            net.at(i).execute(s, net_args.at(i));
    }
    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
                       .count();

    std::cout << "Use time: " << (end - begin) / (times + 0.0) << " ms per iteration." << std::endl;
    s.wait();
}

void cnn_inference_f32(engine::kind engine_kind) {
    int times = 100;

    simple_net(engine_kind, times);
}

int main(int argc, char **argv) {
    return handle_example_errors(
            cnn_inference_f32, parse_engine_kind(argc, argv));
}
