�	�n����?�n����?!�n����?	�6C���5@�6C���5@!�6C���5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�n����?U�	g���?A0��{��?Y��8�~��?*	[d;�O�i@2t
=Iterator::Model::Map::MemoryCacheImpl::ForeverRepeat::BatchV2HP�s�?!m(m�b�P@)�QI��&�?1���0E@:Preprocessing2K
Iterator::Model::Map��B���?!�6��X@)�	/���?1"��a��>@:Preprocessing2�
JIterator::Model::Map::MemoryCacheImpl::ForeverRepeat::BatchV2::TensorSlice�V}��b�?!�uʡTd8@)V}��b�?1�uʡTd8@:Preprocessing2k
4Iterator::Model::Map::MemoryCacheImpl::ForeverRepeat��]i��?!�1��P@)Oʤ�6 [?1���ax��?:Preprocessing2F
Iterator::Model�Az�"�?!      Y@))w���Y?1��䉧�?:Preprocessing2\
%Iterator::Model::Map::MemoryCacheImpl�Y�h9��?!ʪ�By
Q@)}!���S?1��ʻԶ�?:Preprocessing2X
!Iterator::Model::Map::MemoryCache~b���?!ȩ�S Q@)?$D��F?1�����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 21.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2A3.5 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	U�	g���?U�	g���?!U�	g���?      ��!       "      ��!       *      ��!       2	0��{��?0��{��?!0��{��?:      ��!       B      ��!       J	��8�~��?��8�~��?!��8�~��?R      ��!       Z	��8�~��?��8�~��?!��8�~��?JCPU_ONLY2black"�
host�Your program is HIGHLY input-bound because 21.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendationQ
nomoderate"A3.5 % of the total step time sampled is spent on All Others time.:
Refer to the TF2 Profiler FAQ2"CPU: 