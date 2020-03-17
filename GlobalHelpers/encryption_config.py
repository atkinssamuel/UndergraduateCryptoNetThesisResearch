from tensorflow.core.protobuf import rewriter_config_pb2
import tensorflow as tf


def encryption_config(backend="HE_SEAL", encryption_parameters=""):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = (
        rewriter_config_pb2.RewriterConfig.ONE)
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = backend.encode()
    server_config.parameter_map["device_id"].s = b''
    server_config.parameter_map[
        "encryption_parameters"].s = encryption_parameters.encode()
    server_config.parameter_map['enable_client'].s = (str(False)).encode()
    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config
