       �K"	  �~���Abrain.Event:2S?|�     �.y*	�_�~���A"��

n
model/PlaceholderPlaceholder*
dtype0*
shape:��*$
_output_shapes
:��
f
model/Placeholder_1Placeholder*
dtype0*
shape:	�*
_output_shapes
:	�
N
model/ConstConst*
dtype0*
value
B :�*
_output_shapes
: 
O
model/pack/1Const*
dtype0*
value
B :�*
_output_shapes
: 
[

model/packPackmodel/Constmodel/pack/1*
_output_shapes
:*
T0*
N
V
model/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
e
model/zerosFill
model/packmodel/zeros/Const*
T0*(
_output_shapes
:����������
Q
model/pack_1/1Const*
dtype0*
value
B :�*
_output_shapes
: 
_
model/pack_1Packmodel/Constmodel/pack_1/1*
_output_shapes
:*
T0*
N
X
model/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model/zeros_1Fillmodel/pack_1model/zeros_1/Const*
T0*(
_output_shapes
:����������
Q
model/pack_2/1Const*
dtype0*
value
B :�*
_output_shapes
: 
_
model/pack_2Packmodel/Constmodel/pack_2/1*
_output_shapes
:*
T0*
N
X
model/zeros_2/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model/zeros_2Fillmodel/pack_2model/zeros_2/Const*
T0*(
_output_shapes
:����������
Q
model/pack_3/1Const*
dtype0*
value
B :�*
_output_shapes
: 
_
model/pack_3Packmodel/Constmodel/pack_3/1*
_output_shapes
:*
T0*
N
X
model/zeros_3/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model/zeros_3Fillmodel/pack_3model/zeros_3/Const*
T0*(
_output_shapes
:����������
Q
model/pack_4/1Const*
dtype0*
value
B :�*
_output_shapes
: 
_
model/pack_4Packmodel/Constmodel/pack_4/1*
_output_shapes
:*
T0*
N
X
model/zeros_4/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model/zeros_4Fillmodel/pack_4model/zeros_4/Const*
T0*(
_output_shapes
:����������
Q
model/pack_5/1Const*
dtype0*
value
B :�*
_output_shapes
: 
_
model/pack_5Packmodel/Constmodel/pack_5/1*
_output_shapes
:*
T0*
N
X
model/zeros_5/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model/zeros_5Fillmodel/pack_5model/zeros_5/Const*
T0*(
_output_shapes
:����������
W
model/split/split_dimConst*
dtype0*
value	B : *
_output_shapes
: 
~
model/splitSplitmodel/split/split_dimmodel/Placeholder*
	num_split*
T0*$
_output_shapes
:��
g
model/SqueezeSqueezemodel/split*
squeeze_dims
 *
T0* 
_output_shapes
:
��
\
model/dropout/keep_probConst*
dtype0*
valueB
 *�P?*
_output_shapes
: 
P
model/dropout/ShapeShapemodel/Squeeze*
T0*
_output_shapes
:
e
 model/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
e
 model/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
*model/dropout/random_uniform/RandomUniformRandomUniformmodel/dropout/Shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
�
 model/dropout/random_uniform/subSub model/dropout/random_uniform/max model/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
 model/dropout/random_uniform/mulMul*model/dropout/random_uniform/RandomUniform model/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
model/dropout/random_uniformAdd model/dropout/random_uniform/mul model/dropout/random_uniform/min*
T0* 
_output_shapes
:
��
z
model/dropout/addAddmodel/dropout/keep_probmodel/dropout/random_uniform*
T0* 
_output_shapes
:
��
Z
model/dropout/FloorFloormodel/dropout/add*
T0* 
_output_shapes
:
��
R
model/dropout/InvInvmodel/dropout/keep_prob*
T0*
_output_shapes
: 
e
model/dropout/mulMulmodel/Squeezemodel/dropout/Inv*
T0* 
_output_shapes
:
��
m
model/dropout/mul_1Mulmodel/dropout/mulmodel/dropout/Floor*
T0* 
_output_shapes
:
��
�
8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"v  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *�"��*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *�"�;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
dtype0*
seed2 *

seed *
T0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/subSubWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulMulamodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/sub*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Cmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concatConcatCmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat/concat_dimmodel/dropout/mul_1model/zeros_1* 
_output_shapes
:
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
|
:model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/splitSplit:model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dim.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
w
2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1Add2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:22model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel/zeros2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
s
.model/RNN/MultiRNNCell/Cell0/dropout/keep_probConst*
dtype0*
valueB
 *�P?*
_output_shapes
: 
�
*model/RNN/MultiRNNCell/Cell0/dropout/ShapeShape0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
|
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
|
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Amodel/RNN/MultiRNNCell/Cell0/dropout/random_uniform/RandomUniformRandomUniform*model/RNN/MultiRNNCell/Cell0/dropout/Shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell0/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
3model/RNN/MultiRNNCell/Cell0/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell0/dropout/addAdd.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob3model/RNN/MultiRNNCell/Cell0/dropout/random_uniform*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell0/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell0/dropout/add*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell0/dropout/InvInv.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell0/dropout/mulMul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell0/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mul*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0* 
_output_shapes
:
��
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB"J  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *�"��*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *�"�;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
dtype0*
seed2 *

seed *
T0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/subSubWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulMulamodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/sub*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Smodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Cmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatConcatCmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dim*model/RNN/MultiRNNCell/Cell0/dropout/mul_1model/zeros_3* 
_output_shapes
:
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
|
:model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/splitSplit:model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dim.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
w
2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1Add2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:22model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel/zeros_22model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
s
.model/RNN/MultiRNNCell/Cell1/dropout/keep_probConst*
dtype0*
valueB
 *�P?*
_output_shapes
: 
�
*model/RNN/MultiRNNCell/Cell1/dropout/ShapeShape0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
|
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
|
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Amodel/RNN/MultiRNNCell/Cell1/dropout/random_uniform/RandomUniformRandomUniform*model/RNN/MultiRNNCell/Cell1/dropout/Shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell1/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
3model/RNN/MultiRNNCell/Cell1/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell1/dropout/addAdd.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob3model/RNN/MultiRNNCell/Cell1/dropout/random_uniform*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell1/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell1/dropout/add*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell1/dropout/InvInv.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell1/dropout/mulMul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell1/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0* 
_output_shapes
:
��
�
8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB"J  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *�"��*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *�"�;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
dtype0*
seed2 *

seed *
T0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/subSubWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulMulamodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/sub*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Smodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
Cmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concatConcatCmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat/concat_dim*model/RNN/MultiRNNCell/Cell1/dropout/mul_1model/zeros_5* 
_output_shapes
:
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
|
:model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/splitSplit:model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split/split_dim.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
w
2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1Add2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:22model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mulMulmodel/zeros_42model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
1model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
s
.model/RNN/MultiRNNCell/Cell2/dropout/keep_probConst*
dtype0*
valueB
 *�P?*
_output_shapes
: 
�
*model/RNN/MultiRNNCell/Cell2/dropout/ShapeShape0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
|
7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/minConst*
dtype0*
valueB
 *    *
_output_shapes
: 
|
7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/maxConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Amodel/RNN/MultiRNNCell/Cell2/dropout/random_uniform/RandomUniformRandomUniform*model/RNN/MultiRNNCell/Cell2/dropout/Shape*
dtype0*
seed2 *

seed *
T0* 
_output_shapes
:
��
�
7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell2/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
3model/RNN/MultiRNNCell/Cell2/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell2/dropout/random_uniform/min*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell2/dropout/addAdd.model/RNN/MultiRNNCell/Cell2/dropout/keep_prob3model/RNN/MultiRNNCell/Cell2/dropout/random_uniform*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell2/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell2/dropout/add*
T0* 
_output_shapes
:
��
�
(model/RNN/MultiRNNCell/Cell2/dropout/InvInv.model/RNN/MultiRNNCell/Cell2/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell2/dropout/mulMul0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell2/dropout/Inv*
T0* 
_output_shapes
:
��
�
*model/RNN/MultiRNNCell/Cell2/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell2/dropout/mul*model/RNN/MultiRNNCell/Cell2/dropout/Floor*
T0* 
_output_shapes
:
��
Y
model/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
o
model/concatIdentity*model/RNN/MultiRNNCell/Cell2/dropout/mul_1*
T0* 
_output_shapes
:
��
d
model/Reshape/shapeConst*
dtype0*
valueB"�����   *
_output_shapes
:
f
model/ReshapeReshapemodel/concatmodel/Reshape/shape*
T0* 
_output_shapes
:
��
�
model/dense_wVariable*
dtype0*
shape:	�*
	container *
shared_name *
_output_shapes
:	�
�
.model/dense_w/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@model/dense_w*
valueB"�      *
_output_shapes
:
�
,model/dense_w/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *�"��*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *�"�;*
_output_shapes
: 
�
6model/dense_w/Initializer/random_uniform/RandomUniformRandomUniform.model/dense_w/Initializer/random_uniform/shape*
_output_shapes
:	�*
dtype0*
seed2 *

seed *
T0* 
_class
loc:@model/dense_w
�
,model/dense_w/Initializer/random_uniform/subSub,model/dense_w/Initializer/random_uniform/max,model/dense_w/Initializer/random_uniform/min* 
_class
loc:@model/dense_w*
T0*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/mulMul6model/dense_w/Initializer/random_uniform/RandomUniform,model/dense_w/Initializer/random_uniform/sub* 
_class
loc:@model/dense_w*
T0*
_output_shapes
:	�
�
(model/dense_w/Initializer/random_uniformAdd,model/dense_w/Initializer/random_uniform/mul,model/dense_w/Initializer/random_uniform/min* 
_class
loc:@model/dense_w*
T0*
_output_shapes
:	�
�
model/dense_w/AssignAssignmodel/dense_w(model/dense_w/Initializer/random_uniform*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes
:	�
y
model/dense_w/readIdentitymodel/dense_w* 
_class
loc:@model/dense_w*
T0*
_output_shapes
:	�
w
model/dense_bVariable*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
�
.model/dense_b/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@model/dense_b*
valueB:*
_output_shapes
:
�
,model/dense_b/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *�"��*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *�"�;*
_output_shapes
: 
�
6model/dense_b/Initializer/random_uniform/RandomUniformRandomUniform.model/dense_b/Initializer/random_uniform/shape*
_output_shapes
:*
dtype0*
seed2 *

seed *
T0* 
_class
loc:@model/dense_b
�
,model/dense_b/Initializer/random_uniform/subSub,model/dense_b/Initializer/random_uniform/max,model/dense_b/Initializer/random_uniform/min* 
_class
loc:@model/dense_b*
T0*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/mulMul6model/dense_b/Initializer/random_uniform/RandomUniform,model/dense_b/Initializer/random_uniform/sub* 
_class
loc:@model/dense_b*
T0*
_output_shapes
:
�
(model/dense_b/Initializer/random_uniformAdd,model/dense_b/Initializer/random_uniform/mul,model/dense_b/Initializer/random_uniform/min* 
_class
loc:@model/dense_b*
T0*
_output_shapes
:
�
model/dense_b/AssignAssignmodel/dense_b(model/dense_b/Initializer/random_uniform*
validate_shape(* 
_class
loc:@model/dense_b*
use_locking(*
T0*
_output_shapes
:
t
model/dense_b/readIdentitymodel/dense_b* 
_class
loc:@model/dense_b*
T0*
_output_shapes
:
�
model/MatMulMatMulmodel/Reshapemodel/dense_w/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	�
\
	model/addAddmodel/MatMulmodel/dense_b/read*
T0*
_output_shapes
:	�
f
model/Reshape_1/shapeConst*
dtype0*
valueB"   �  *
_output_shapes
:
f
model/Reshape_1Reshape	model/addmodel/Reshape_1/shape*
T0*
_output_shapes
:	�
`
	model/SubSubmodel/Placeholder_1model/Reshape_1*
T0*
_output_shapes
:	�
K
model/SquareSquare	model/Sub*
T0*
_output_shapes
:	�
A

model/RankRankmodel/Square*
T0*
_output_shapes
: 
S
model/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
S
model/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
b
model/rangeRangemodel/range/start
model/Rankmodel/range/delta*
_output_shapes
:
]
	model/SumSummodel/Squaremodel/range*
T0*
	keep_dims( *
_output_shapes
: 
J
model/Rank_1Rankmodel/Placeholder_1*
T0*
_output_shapes
: 
U
model/range_1/startConst*
dtype0*
value	B : *
_output_shapes
: 
U
model/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
j
model/range_1Rangemodel/range_1/startmodel/Rank_1model/range_1/delta*
_output_shapes
:
h

model/MeanMeanmodel/Placeholder_1model/range_1*
T0*
	keep_dims( *
_output_shapes
: 
]
model/Sub_1Submodel/Placeholder_1
model/Mean*
T0*
_output_shapes
:	�
O
model/Square_1Squaremodel/Sub_1*
T0*
_output_shapes
:	�
E
model/Rank_2Rankmodel/Square_1*
T0*
_output_shapes
: 
U
model/range_2/startConst*
dtype0*
value	B : *
_output_shapes
: 
U
model/range_2/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
j
model/range_2Rangemodel/range_2/startmodel/Rank_2model/range_2/delta*
_output_shapes
:
c
model/Sum_1Summodel/Square_1model/range_2*
T0*
	keep_dims( *
_output_shapes
: 
I
	model/DivDiv	model/Summodel/Sum_1*
T0*
_output_shapes
: 
R
model/Sub_2/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
M
model/Sub_2Submodel/Sub_2/x	model/Div*
T0*
_output_shapes
: 
b
model/Sub_3Submodel/Placeholder_1model/Reshape_1*
T0*
_output_shapes
:	�
O
model/Square_2Squaremodel/Sub_3*
T0*
_output_shapes
:	�
E
model/Rank_3Rankmodel/Square_2*
T0*
_output_shapes
: 
U
model/range_3/startConst*
dtype0*
value	B : *
_output_shapes
: 
U
model/range_3/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
j
model/range_3Rangemodel/range_3/startmodel/Rank_3model/range_3/delta*
_output_shapes
:
e
model/Mean_1Meanmodel/Square_2model/range_3*
T0*
	keep_dims( *
_output_shapes
: 
v
model/zeros_6Const*
dtype0*'
valueB��*    *(
_output_shapes
:��
�
model/VariableVariable*
dtype0*
shape:��*
	container *
shared_name *(
_output_shapes
:��
�
model/Variable/AssignAssignmodel/Variablemodel/zeros_6*
validate_shape(*!
_class
loc:@model/Variable*
use_locking(*
T0*(
_output_shapes
:��
�
model/Variable/readIdentitymodel/Variable*!
_class
loc:@model/Variable*
T0*(
_output_shapes
:��
�
model/Assign/value/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/Assign/value/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/Assign/value/2Pack0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/Assign/valuePackmodel/Assign/value/0model/Assign/value/1model/Assign/value/2*(
_output_shapes
:��*
T0*
N
�
model/AssignAssignmodel/Variablemodel/Assign/value*
validate_shape(*!
_class
loc:@model/Variable*
use_locking( *
T0*(
_output_shapes
:��
c
model/Variable_1/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
r
model/Variable_1Variable*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
�
model/Variable_1/AssignAssignmodel/Variable_1model/Variable_1/initial_value*
validate_shape(*#
_class
loc:@model/Variable_1*
use_locking(*
T0*
_output_shapes
: 
y
model/Variable_1/readIdentitymodel/Variable_1*#
_class
loc:@model/Variable_1*
T0*
_output_shapes
: 
O
model/gradients/ShapeShapemodel/Mean_1*
T0*
_output_shapes
: 
Z
model/gradients/ConstConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
k
model/gradients/FillFillmodel/gradients/Shapemodel/gradients/Const*
T0*
_output_shapes
: 
e
'model/gradients/model/Mean_1_grad/ShapeShapemodel/Square_2*
T0*
_output_shapes
:
x
&model/gradients/model/Mean_1_grad/SizeSize'model/gradients/model/Mean_1_grad/Shape*
T0*
_output_shapes
: 
�
%model/gradients/model/Mean_1_grad/addAddmodel/range_3&model/gradients/model/Mean_1_grad/Size*
T0*
_output_shapes
:
�
%model/gradients/model/Mean_1_grad/modMod%model/gradients/model/Mean_1_grad/add&model/gradients/model/Mean_1_grad/Size*
T0*
_output_shapes
:
~
)model/gradients/model/Mean_1_grad/Shape_1Shape%model/gradients/model/Mean_1_grad/mod*
T0*
_output_shapes
:
o
-model/gradients/model/Mean_1_grad/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
o
-model/gradients/model/Mean_1_grad/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
'model/gradients/model/Mean_1_grad/rangeRange-model/gradients/model/Mean_1_grad/range/start&model/gradients/model/Mean_1_grad/Size-model/gradients/model/Mean_1_grad/range/delta*
_output_shapes
:
n
,model/gradients/model/Mean_1_grad/Fill/valueConst*
dtype0*
value	B :*
_output_shapes
: 
�
&model/gradients/model/Mean_1_grad/FillFill)model/gradients/model/Mean_1_grad/Shape_1,model/gradients/model/Mean_1_grad/Fill/value*
T0*
_output_shapes
:
�
/model/gradients/model/Mean_1_grad/DynamicStitchDynamicStitch'model/gradients/model/Mean_1_grad/range%model/gradients/model/Mean_1_grad/mod'model/gradients/model/Mean_1_grad/Shape&model/gradients/model/Mean_1_grad/Fill*#
_output_shapes
:���������*
T0*
N
m
+model/gradients/model/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
)model/gradients/model/Mean_1_grad/MaximumMaximum/model/gradients/model/Mean_1_grad/DynamicStitch+model/gradients/model/Mean_1_grad/Maximum/y*
T0*#
_output_shapes
:���������
�
*model/gradients/model/Mean_1_grad/floordivDiv'model/gradients/model/Mean_1_grad/Shape)model/gradients/model/Mean_1_grad/Maximum*
T0*
_output_shapes
:
�
)model/gradients/model/Mean_1_grad/ReshapeReshapemodel/gradients/Fill/model/gradients/model/Mean_1_grad/DynamicStitch*
T0*
_output_shapes
:
�
&model/gradients/model/Mean_1_grad/TileTile)model/gradients/model/Mean_1_grad/Reshape*model/gradients/model/Mean_1_grad/floordiv*
T0*0
_output_shapes
:������������������
g
)model/gradients/model/Mean_1_grad/Shape_2Shapemodel/Square_2*
T0*
_output_shapes
:
c
)model/gradients/model/Mean_1_grad/Shape_3Shapemodel/Mean_1*
T0*
_output_shapes
: 
z
&model/gradients/model/Mean_1_grad/RankRank)model/gradients/model/Mean_1_grad/Shape_2*
T0*
_output_shapes
: 
q
/model/gradients/model/Mean_1_grad/range_1/startConst*
dtype0*
value	B : *
_output_shapes
: 
q
/model/gradients/model/Mean_1_grad/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
)model/gradients/model/Mean_1_grad/range_1Range/model/gradients/model/Mean_1_grad/range_1/start&model/gradients/model/Mean_1_grad/Rank/model/gradients/model/Mean_1_grad/range_1/delta*
_output_shapes
:
�
&model/gradients/model/Mean_1_grad/ProdProd)model/gradients/model/Mean_1_grad/Shape_2)model/gradients/model/Mean_1_grad/range_1*
T0*
	keep_dims( *
_output_shapes
: 
|
(model/gradients/model/Mean_1_grad/Rank_1Rank)model/gradients/model/Mean_1_grad/Shape_3*
T0*
_output_shapes
: 
q
/model/gradients/model/Mean_1_grad/range_2/startConst*
dtype0*
value	B : *
_output_shapes
: 
q
/model/gradients/model/Mean_1_grad/range_2/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
)model/gradients/model/Mean_1_grad/range_2Range/model/gradients/model/Mean_1_grad/range_2/start(model/gradients/model/Mean_1_grad/Rank_1/model/gradients/model/Mean_1_grad/range_2/delta*
_output_shapes
:
�
(model/gradients/model/Mean_1_grad/Prod_1Prod)model/gradients/model/Mean_1_grad/Shape_3)model/gradients/model/Mean_1_grad/range_2*
T0*
	keep_dims( *
_output_shapes
: 
o
-model/gradients/model/Mean_1_grad/Maximum_1/yConst*
dtype0*
value	B :*
_output_shapes
: 
�
+model/gradients/model/Mean_1_grad/Maximum_1Maximum(model/gradients/model/Mean_1_grad/Prod_1-model/gradients/model/Mean_1_grad/Maximum_1/y*
T0*
_output_shapes
: 
�
,model/gradients/model/Mean_1_grad/floordiv_1Div&model/gradients/model/Mean_1_grad/Prod+model/gradients/model/Mean_1_grad/Maximum_1*
T0*
_output_shapes
: 
�
&model/gradients/model/Mean_1_grad/CastCast,model/gradients/model/Mean_1_grad/floordiv_1*

DstT0*

SrcT0*
_output_shapes
: 
�
)model/gradients/model/Mean_1_grad/truedivDiv&model/gradients/model/Mean_1_grad/Tile&model/gradients/model/Mean_1_grad/Cast*
T0*
_output_shapes
:	�
�
)model/gradients/model/Square_2_grad/mul/xConst*^model/gradients/model/Mean_1_grad/truediv*
dtype0*
valueB
 *   @*
_output_shapes
: 
�
'model/gradients/model/Square_2_grad/mulMul)model/gradients/model/Square_2_grad/mul/xmodel/Sub_3*
T0*
_output_shapes
:	�
�
)model/gradients/model/Square_2_grad/mul_1Mul)model/gradients/model/Mean_1_grad/truediv'model/gradients/model/Square_2_grad/mul*
T0*
_output_shapes
:	�
i
&model/gradients/model/Sub_3_grad/ShapeShapemodel/Placeholder_1*
T0*
_output_shapes
:
g
(model/gradients/model/Sub_3_grad/Shape_1Shapemodel/Reshape_1*
T0*
_output_shapes
:
�
6model/gradients/model/Sub_3_grad/BroadcastGradientArgsBroadcastGradientArgs&model/gradients/model/Sub_3_grad/Shape(model/gradients/model/Sub_3_grad/Shape_1*2
_output_shapes 
:���������:���������
�
$model/gradients/model/Sub_3_grad/SumSum)model/gradients/model/Square_2_grad/mul_16model/gradients/model/Sub_3_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
(model/gradients/model/Sub_3_grad/ReshapeReshape$model/gradients/model/Sub_3_grad/Sum&model/gradients/model/Sub_3_grad/Shape*
T0*
_output_shapes
:	�
�
&model/gradients/model/Sub_3_grad/Sum_1Sum)model/gradients/model/Square_2_grad/mul_18model/gradients/model/Sub_3_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
v
$model/gradients/model/Sub_3_grad/NegNeg&model/gradients/model/Sub_3_grad/Sum_1*
T0*
_output_shapes
:
�
*model/gradients/model/Sub_3_grad/Reshape_1Reshape$model/gradients/model/Sub_3_grad/Neg(model/gradients/model/Sub_3_grad/Shape_1*
T0*
_output_shapes
:	�
c
*model/gradients/model/Reshape_1_grad/ShapeShape	model/add*
T0*
_output_shapes
:
�
,model/gradients/model/Reshape_1_grad/ReshapeReshape*model/gradients/model/Sub_3_grad/Reshape_1*model/gradients/model/Reshape_1_grad/Shape*
T0*
_output_shapes
:	�
`
$model/gradients/model/add_grad/ShapeShapemodel/MatMul*
T0*
_output_shapes
:
h
&model/gradients/model/add_grad/Shape_1Shapemodel/dense_b/read*
T0*
_output_shapes
:
�
4model/gradients/model/add_grad/BroadcastGradientArgsBroadcastGradientArgs$model/gradients/model/add_grad/Shape&model/gradients/model/add_grad/Shape_1*2
_output_shapes 
:���������:���������
�
"model/gradients/model/add_grad/SumSum,model/gradients/model/Reshape_1_grad/Reshape4model/gradients/model/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
&model/gradients/model/add_grad/ReshapeReshape"model/gradients/model/add_grad/Sum$model/gradients/model/add_grad/Shape*
T0*
_output_shapes
:	�
�
$model/gradients/model/add_grad/Sum_1Sum,model/gradients/model/Reshape_1_grad/Reshape6model/gradients/model/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
(model/gradients/model/add_grad/Reshape_1Reshape$model/gradients/model/add_grad/Sum_1&model/gradients/model/add_grad/Shape_1*
T0*
_output_shapes
:
�
(model/gradients/model/MatMul_grad/MatMulMatMul&model/gradients/model/add_grad/Reshapemodel/dense_w/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
*model/gradients/model/MatMul_grad/MatMul_1MatMulmodel/Reshape&model/gradients/model/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	�
d
(model/gradients/model/Reshape_grad/ShapeShapemodel/concat*
T0*
_output_shapes
:
�
*model/gradients/model/Reshape_grad/ReshapeReshape(model/gradients/model/MatMul_grad/MatMul(model/gradients/model/Reshape_grad/Shape*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/ShapeShape(model/RNN/MultiRNNCell/Cell2/dropout/mul*
T0*
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Shape_1Shape*model/RNN/MultiRNNCell/Cell2/dropout/Floor*
T0*
_output_shapes
:
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/ShapeGmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/mulMul*model/gradients/model/Reshape_grad/Reshape*model/RNN/MultiRNNCell/Cell2/dropout/Floor*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/SumSumCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/mulUmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/ReshapeReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/SumEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell2/dropout/mul*model/gradients/model/Reshape_grad/Reshape*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Sum_1SumEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/mul_1Wmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Reshape_1ReshapeEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Sum_1Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/ShapeShape0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Shape_1Shape(model/RNN/MultiRNNCell/Cell2/dropout/Inv*
T0*
_output_shapes
: 
�
Smodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/ShapeEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/mulMulGmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Reshape(model/RNN/MultiRNNCell/Cell2/dropout/Inv*
T0* 
_output_shapes
:
��
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/SumSumAmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/ReshapeReshapeAmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/SumCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Shape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_1_grad/Reshape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Sum_1SumCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/mul_1Umodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Reshape_1ReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Sum_1Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/ShapeShape1model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Shape_1Shape4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/mulMulEmodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Reshape4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell2/dropout/mul_grad/Reshape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/subSubLmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/sub/xMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/Square*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/sub/x4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/ShapeShape.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Shape_1Shape0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/SumSumJmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Sum_1SumJmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1_grad/mul]model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/ShapeShapemodel/zeros_4*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Reshape2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/SumSumGmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/mulYmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Shape*
T0*(
_output_shapes
:����������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/mul_1Mulmodel/zeros_4Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Sum_1SumImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/ShapeShape4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Shape_1Shape/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Reshape_1/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/subSubMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/sub/x2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/sub*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/mul*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/sub/x4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/subSubJmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/sub/xKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/Square*
T0* 
_output_shapes
:
��
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/sub*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/ShapeShape2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:2*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1/y*
T0*
_output_shapes
: 
�
[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/SumSumMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Sum_1SumMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/Shape_1*
T0*
_output_shapes
: 
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split_grad/concatConcat:model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split/split_dimOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1_grad/mul_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_grad/mulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1_grad/ReshapeOmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2_grad/mul_1* 
_output_shapes
:
��*
T0*
N
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/ShapeShape8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Shape_1Shape;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/read*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/SumSumLmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split_grad/concatYmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Shape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Sum_1SumLmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split_grad/concat[model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Shape_1*
T0*
_output_shapes	
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ShapeNShapeN*model/RNN/MultiRNNCell/Cell1/dropout/mul_1model/zeros_5* 
_output_shapes
::*
T0*
N
�
Zmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ConcatOffsetConcatOffsetCmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat/concat_dimTmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ShapeNVmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ShapeN:1* 
_output_shapes
::*
N
�
Smodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/SliceSliceTmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMulZmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ConcatOffsetTmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:
��
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/Slice_1SliceTmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul\model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ConcatOffset:1Vmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:����������
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/ShapeShape(model/RNN/MultiRNNCell/Cell1/dropout/mul*
T0*
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape_1Shape*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0*
_output_shapes
:
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/ShapeGmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mulMulSmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/Slice*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/SumSumCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mulUmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/ReshapeReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/SumEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat_grad/Slice*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Sum_1SumEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Wmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape_1ReshapeEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Sum_1Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/ShapeShape0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Shape_1Shape(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0*
_output_shapes
: 
�
Smodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/ShapeEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mulMulGmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0* 
_output_shapes
:
��
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/SumSumAmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/ReshapeReshapeAmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/SumCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Shape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Sum_1SumCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mul_1Umodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape_1ReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Sum_1Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ShapeShape1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape_1Shape4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mulMulEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/subSubLmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub/xMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/Square*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub/x4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/ShapeShape.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Shape_1Shape0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/SumSumJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Sum_1SumJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mul]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/ShapeShapemodel/zeros_2*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/SumSumGmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mulYmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape*
T0*(
_output_shapes
:����������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1Mulmodel/zeros_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Sum_1SumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ShapeShape4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape_1Shape/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/subSubMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub/x2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub/x4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/subSubJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub/xKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/Square*
T0* 
_output_shapes
:
��
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/ShapeShape2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:2*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1/y*
T0*
_output_shapes
: 
�
[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/SumSumMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Sum_1SumMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/Shape_1*
T0*
_output_shapes
: 
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split_grad/concatConcat:model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dimOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/mulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1_grad/ReshapeOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul_1* 
_output_shapes
:
��*
T0*
N
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/ShapeShape8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Shape_1Shape;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/SumSumLmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split_grad/concatYmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Shape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Sum_1SumLmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split_grad/concat[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Shape_1*
T0*
_output_shapes	
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeNShapeN*model/RNN/MultiRNNCell/Cell0/dropout/mul_1model/zeros_3* 
_output_shapes
::*
T0*
N
�
Zmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffsetConcatOffsetCmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dimTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeNVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN:1* 
_output_shapes
::*
N
�
Smodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/SliceSliceTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMulZmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffsetTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN*
Index0*
T0* 
_output_shapes
:
��
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice_1SliceTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul\model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffset:1Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN:1*
Index0*
T0*(
_output_shapes
:����������
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/ShapeShape(model/RNN/MultiRNNCell/Cell0/dropout/mul*
T0*
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape_1Shape*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0*
_output_shapes
:
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/ShapeGmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mulMulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/SumSumCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mulUmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/ReshapeReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/SumEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice*
T0* 
_output_shapes
:
��
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Sum_1SumEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Wmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape_1ReshapeEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Sum_1Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/ShapeShape0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*
T0*
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Shape_1Shape(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0*
_output_shapes
: 
�
Smodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/ShapeEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mulMulGmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0* 
_output_shapes
:
��
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/SumSumAmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/ReshapeReshapeAmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/SumCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Shape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape*
T0* 
_output_shapes
:
��
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Sum_1SumCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mul_1Umodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape_1ReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Sum_1Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Shape_1*
T0*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ShapeShape1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape_1Shape4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mulMulEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/subSubLmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub/xMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/Square*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub/x4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/ShapeShape.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Shape_1Shape0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/SumSumJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Sum_1SumJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mul]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/ShapeShapemodel/zeros*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/SumSumGmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mulYmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape*
T0*(
_output_shapes
:����������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1Mulmodel/zerosMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Sum_1SumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ShapeShape4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape_1Shape/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:
�
[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/subSubMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub/x2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/subSubOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub/x4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1*
T0* 
_output_shapes
:
��
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub*
T0* 
_output_shapes
:
��
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1*
T0* 
_output_shapes
:
��
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub/xConstP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/subSubJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub/xKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/Square*
T0* 
_output_shapes
:
��
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/ShapeShape2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:2*
T0*
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Shape_1Shape2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1/y*
T0*
_output_shapes
: 
�
[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/ShapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/SumSumMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Shape*
T0* 
_output_shapes
:
��
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Sum_1SumMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/Shape_1*
T0*
_output_shapes
: 
�
Lmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split_grad/concatConcat:model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dimOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/mulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1_grad/ReshapeOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul_1* 
_output_shapes
:
��*
T0*
N
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/ShapeShape8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul*
T0*
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Shape_1Shape;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0*
_output_shapes
:
�
Ymodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/BroadcastGradientArgsBroadcastGradientArgsImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/ShapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Shape_1*2
_output_shapes 
:���������:���������
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/SumSumLmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split_grad/concatYmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Shape*
T0* 
_output_shapes
:
��
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Sum_1SumLmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split_grad/concat[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Shape_1*
T0*
_output_shapes	
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
�
model/global_norm/L2LossL2LossVmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_1L2LossMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_2L2LossVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_3L2LossMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_4L2LossVmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_5L2LossMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_6L2Loss*model/gradients/model/MatMul_grad/MatMul_1*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_7L2Loss(model/gradients/model/add_grad/Reshape_1*;
_class1
/-loc:@model/gradients/model/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
model/global_norm/packPackmodel/global_norm/L2Lossmodel/global_norm/L2Loss_1model/global_norm/L2Loss_2model/global_norm/L2Loss_3model/global_norm/L2Loss_4model/global_norm/L2Loss_5model/global_norm/L2Loss_6model/global_norm/L2Loss_7*
_output_shapes
:*
T0*
N
W
model/global_norm/RankRankmodel/global_norm/pack*
T0*
_output_shapes
: 
_
model/global_norm/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
_
model/global_norm/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
�
model/global_norm/rangeRangemodel/global_norm/range/startmodel/global_norm/Rankmodel/global_norm/range/delta*
_output_shapes
:

model/global_norm/SumSummodel/global_norm/packmodel/global_norm/range*
T0*
	keep_dims( *
_output_shapes
: 
\
model/global_norm/ConstConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
m
model/global_norm/mulMulmodel/global_norm/Summodel/global_norm/Const*
T0*
_output_shapes
: 
]
model/global_norm/global_normSqrtmodel/global_norm/mul*
T0*
_output_shapes
: 
h
#model/clip_by_global_norm/truediv/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
!model/clip_by_global_norm/truedivDiv#model/clip_by_global_norm/truediv/xmodel/global_norm/global_norm*
T0*
_output_shapes
: 
d
model/clip_by_global_norm/ConstConst*
dtype0*
valueB
 *���=*
_output_shapes
: 
�
!model/clip_by_global_norm/MinimumMinimum!model/clip_by_global_norm/truedivmodel/clip_by_global_norm/Const*
T0*
_output_shapes
: 
d
model/clip_by_global_norm/mul/xConst*
dtype0*
valueB
 *  pA*
_output_shapes
: 
�
model/clip_by_global_norm/mulMulmodel/clip_by_global_norm/mul/x!model/clip_by_global_norm/Minimum*
T0*
_output_shapes
: 
�
model/clip_by_global_norm/mul_1MulVmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_2MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_1Identitymodel/clip_by_global_norm/mul_2*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_3MulVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_2Identitymodel/clip_by_global_norm/mul_3*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_4MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_3Identitymodel/clip_by_global_norm/mul_4*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_5MulVmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_4Identitymodel/clip_by_global_norm/mul_5*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_6MulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_5Identitymodel/clip_by_global_norm/mul_6*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_7Mul*model/gradients/model/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_6Identitymodel/clip_by_global_norm/mul_7*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	�
�
model/clip_by_global_norm/mul_8Mul(model/gradients/model/add_grad/Reshape_1model/clip_by_global_norm/mul*;
_class1
/-loc:@model/gradients/model/add_grad/Reshape_1*
T0*
_output_shapes
:
�
6model/clip_by_global_norm/model/clip_by_global_norm/_7Identitymodel/clip_by_global_norm/mul_8*;
_class1
/-loc:@model/gradients/model/add_grad/Reshape_1*
T0*
_output_shapes
:
�
model/beta1_power/initial_valueConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *fff?*
_output_shapes
: 
�
model/beta1_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
model/beta1_power/AssignAssignmodel/beta1_powermodel/beta1_power/initial_value*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0*
_output_shapes
: 
�
model/beta1_power/readIdentitymodel/beta1_power*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
model/beta2_power/initial_valueConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *w�?*
_output_shapes
: 
�
model/beta2_powerVariable*
	container *
_output_shapes
: *
dtype0*
shape: *K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
model/beta2_power/AssignAssignmodel/beta2_powermodel/beta2_power/initial_value*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0*
_output_shapes
: 
�
model/beta2_power/readIdentitymodel/beta2_power*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
f
model/zeros_7Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adammodel/zeros_7*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_8Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Lmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/AssignAssignEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1model/zeros_8*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
\
model/zeros_9Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
shared_name 
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/AssignAssignAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adammodel/zeros_9*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_10Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/zeros_10*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
g
model/zeros_11Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adammodel/zeros_11*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_12Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Lmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/AssignAssignEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/zeros_12*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_13Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
shared_name 
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/AssignAssignAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adammodel/zeros_13*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_14Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/zeros_14*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
g
model/zeros_15Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adammodel/zeros_15*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Hmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_16Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Lmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1/AssignAssignEmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1model/zeros_16*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Jmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_17Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
shared_name 
�
Hmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam/AssignAssignAmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adammodel/zeros_17*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_18Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1model/zeros_18*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
e
model/zeros_19Const*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
model/model/dense_w/AdamVariable*
	container *
_output_shapes
:	�*
dtype0*
shape:	�* 
_class
loc:@model/dense_w*
shared_name 
�
model/model/dense_w/Adam/AssignAssignmodel/model/dense_w/Adammodel/zeros_19*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes
:	�
�
model/model/dense_w/Adam/readIdentitymodel/model/dense_w/Adam* 
_class
loc:@model/dense_w*
T0*
_output_shapes
:	�
e
model/zeros_20Const*
dtype0*
valueB	�*    *
_output_shapes
:	�
�
model/model/dense_w/Adam_1Variable*
	container *
_output_shapes
:	�*
dtype0*
shape:	�* 
_class
loc:@model/dense_w*
shared_name 
�
!model/model/dense_w/Adam_1/AssignAssignmodel/model/dense_w/Adam_1model/zeros_20*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes
:	�
�
model/model/dense_w/Adam_1/readIdentitymodel/model/dense_w/Adam_1* 
_class
loc:@model/dense_w*
T0*
_output_shapes
:	�
[
model/zeros_21Const*
dtype0*
valueB*    *
_output_shapes
:
�
model/model/dense_b/AdamVariable*
	container *
_output_shapes
:*
dtype0*
shape:* 
_class
loc:@model/dense_b*
shared_name 
�
model/model/dense_b/Adam/AssignAssignmodel/model/dense_b/Adammodel/zeros_21*
validate_shape(* 
_class
loc:@model/dense_b*
use_locking(*
T0*
_output_shapes
:
�
model/model/dense_b/Adam/readIdentitymodel/model/dense_b/Adam* 
_class
loc:@model/dense_b*
T0*
_output_shapes
:
[
model/zeros_22Const*
dtype0*
valueB*    *
_output_shapes
:
�
model/model/dense_b/Adam_1Variable*
	container *
_output_shapes
:*
dtype0*
shape:* 
_class
loc:@model/dense_b*
shared_name 
�
!model/model/dense_b/Adam_1/AssignAssignmodel/model/dense_b/Adam_1model/zeros_22*
validate_shape(* 
_class
loc:@model/dense_b*
use_locking(*
T0*
_output_shapes
:
�
model/model/dense_b/Adam_1/readIdentitymodel/model/dense_b/Adam_1* 
_class
loc:@model/dense_b*
T0*
_output_shapes
:
U
model/Adam/beta1Const*
dtype0*
valueB
 *fff?*
_output_shapes
: 
U
model/Adam/beta2Const*
dtype0*
valueB
 *w�?*
_output_shapes
: 
W
model/Adam/epsilonConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_2*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_3*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_4*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_5*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
)model/Adam/update_model/dense_w/ApplyAdam	ApplyAdammodel/dense_wmodel/model/dense_w/Adammodel/model/dense_w/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_6* 
_class
loc:@model/dense_w*
use_locking( *
T0*
_output_shapes
:	�
�
)model/Adam/update_model/dense_b/ApplyAdam	ApplyAdammodel/dense_bmodel/model/dense_b/Adammodel/model/dense_b/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_7* 
_class
loc:@model/dense_b*
use_locking( *
T0*
_output_shapes
:
�
model/Adam/mulMulmodel/beta1_power/readmodel/Adam/beta1U^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
model/Adam/AssignAssignmodel/beta1_powermodel/Adam/mul*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0*
_output_shapes
: 
�
model/Adam/mul_1Mulmodel/beta2_power/readmodel/Adam/beta2U^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
: 
�
model/Adam/Assign_1Assignmodel/beta2_powermodel/Adam/mul_1*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0*
_output_shapes
: 
�

model/AdamNoOpU^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam^model/Adam/Assign^model/Adam/Assign_1
k
model/ScalarSummary/tagsConst*
dtype0*#
valueB Bmean squared error*
_output_shapes
: 
m
model/ScalarSummaryScalarSummarymodel/ScalarSummary/tagsmodel/Mean_1*
T0*
_output_shapes
: 
d
model/ScalarSummary_1/tagsConst*
dtype0*
valueB B	r-squared*
_output_shapes
: 
p
model/ScalarSummary_1ScalarSummarymodel/ScalarSummary_1/tagsmodel/Sub_2*
T0*
_output_shapes
: 
a
model/HistogramSummary/tagConst*
dtype0*
valueB Bstates*
_output_shapes
: 
�
model/HistogramSummary/values/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/HistogramSummary/values/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/HistogramSummary/values/2Pack0model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model/HistogramSummary/valuesPackmodel/HistogramSummary/values/0model/HistogramSummary/values/1model/HistogramSummary/values/2*(
_output_shapes
:��*
T0*
N
�
model/HistogramSummaryHistogramSummarymodel/HistogramSummary/tagmodel/HistogramSummary/values*
T0*
_output_shapes
: 
h
model/HistogramSummary_1/tagConst*
dtype0*
valueB Bpredictions*
_output_shapes
: 
|
model/HistogramSummary_1HistogramSummarymodel/HistogramSummary_1/tagmodel/Reshape_1*
T0*
_output_shapes
: 
�
model/MergeSummary/MergeSummaryMergeSummarymodel/ScalarSummarymodel/ScalarSummary_1model/HistogramSummarymodel/HistogramSummary_1*
_output_shapes
: *
N
p
model_1/PlaceholderPlaceholder*
dtype0*
shape:��*$
_output_shapes
:��
h
model_1/Placeholder_1Placeholder*
dtype0*
shape:	�*
_output_shapes
:	�
P
model_1/ConstConst*
dtype0*
value
B :�*
_output_shapes
: 
Q
model_1/pack/1Const*
dtype0*
value
B :�*
_output_shapes
: 
a
model_1/packPackmodel_1/Constmodel_1/pack/1*
_output_shapes
:*
T0*
N
X
model_1/zeros/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
k
model_1/zerosFillmodel_1/packmodel_1/zeros/Const*
T0*(
_output_shapes
:����������
S
model_1/pack_1/1Const*
dtype0*
value
B :�*
_output_shapes
: 
e
model_1/pack_1Packmodel_1/Constmodel_1/pack_1/1*
_output_shapes
:*
T0*
N
Z
model_1/zeros_1/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
model_1/zeros_1Fillmodel_1/pack_1model_1/zeros_1/Const*
T0*(
_output_shapes
:����������
S
model_1/pack_2/1Const*
dtype0*
value
B :�*
_output_shapes
: 
e
model_1/pack_2Packmodel_1/Constmodel_1/pack_2/1*
_output_shapes
:*
T0*
N
Z
model_1/zeros_2/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
model_1/zeros_2Fillmodel_1/pack_2model_1/zeros_2/Const*
T0*(
_output_shapes
:����������
S
model_1/pack_3/1Const*
dtype0*
value
B :�*
_output_shapes
: 
e
model_1/pack_3Packmodel_1/Constmodel_1/pack_3/1*
_output_shapes
:*
T0*
N
Z
model_1/zeros_3/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
model_1/zeros_3Fillmodel_1/pack_3model_1/zeros_3/Const*
T0*(
_output_shapes
:����������
S
model_1/pack_4/1Const*
dtype0*
value
B :�*
_output_shapes
: 
e
model_1/pack_4Packmodel_1/Constmodel_1/pack_4/1*
_output_shapes
:*
T0*
N
Z
model_1/zeros_4/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
model_1/zeros_4Fillmodel_1/pack_4model_1/zeros_4/Const*
T0*(
_output_shapes
:����������
S
model_1/pack_5/1Const*
dtype0*
value
B :�*
_output_shapes
: 
e
model_1/pack_5Packmodel_1/Constmodel_1/pack_5/1*
_output_shapes
:*
T0*
N
Z
model_1/zeros_5/ConstConst*
dtype0*
valueB
 *    *
_output_shapes
: 
q
model_1/zeros_5Fillmodel_1/pack_5model_1/zeros_5/Const*
T0*(
_output_shapes
:����������
Y
model_1/split/split_dimConst*
dtype0*
value	B : *
_output_shapes
: 
�
model_1/splitSplitmodel_1/split/split_dimmodel_1/Placeholder*
	num_split*
T0*$
_output_shapes
:��
k
model_1/SqueezeSqueezemodel_1/split*
squeeze_dims
 *
T0* 
_output_shapes
:
��
�
Emodel_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concatConcatEmodel_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat/concat_dimmodel_1/Squeezemodel_1/zeros_1* 
_output_shapes
:
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
~
<model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/splitSplit<model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dim0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
y
4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1Add4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:24model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel_1/zeros4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
1model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Emodel_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatConcatEmodel_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dim2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2model_1/zeros_3* 
_output_shapes
:
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
~
<model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/splitSplit<model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dim0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
y
4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1Add4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:24model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel_1/zeros_24model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
1model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Emodel_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concatConcatEmodel_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat/concat_dim2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2model_1/zeros_5* 
_output_shapes
:
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
~
<model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/splitSplit<model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split/split_dim0model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add*
	num_split*
T0*D
_output_shapes2
0:
��:
��:
��:
��
y
4model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1Add4model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:24model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1/y*
T0* 
_output_shapes
:
��
�
4model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_1*
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mulMulmodel_1/zeros_44model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split*
T0* 
_output_shapes
:
��
�
1model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:1*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_1*
T0* 
_output_shapes
:
��
�
3model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_2*
T0* 
_output_shapes
:
��
�
6model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/split:3*
T0* 
_output_shapes
:
��
�
2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Sigmoid_2*
T0* 
_output_shapes
:
��
[
model_1/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
y
model_1/concatIdentity2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*
T0* 
_output_shapes
:
��
f
model_1/Reshape/shapeConst*
dtype0*
valueB"�����   *
_output_shapes
:
l
model_1/ReshapeReshapemodel_1/concatmodel_1/Reshape/shape*
T0* 
_output_shapes
:
��
�
model_1/MatMulMatMulmodel_1/Reshapemodel/dense_w/read*
transpose_b( *
transpose_a( *
T0*
_output_shapes
:	�
`
model_1/addAddmodel_1/MatMulmodel/dense_b/read*
T0*
_output_shapes
:	�
h
model_1/Reshape_1/shapeConst*
dtype0*
valueB"   �  *
_output_shapes
:
l
model_1/Reshape_1Reshapemodel_1/addmodel_1/Reshape_1/shape*
T0*
_output_shapes
:	�
f
model_1/SubSubmodel_1/Placeholder_1model_1/Reshape_1*
T0*
_output_shapes
:	�
O
model_1/SquareSquaremodel_1/Sub*
T0*
_output_shapes
:	�
E
model_1/RankRankmodel_1/Square*
T0*
_output_shapes
: 
U
model_1/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
U
model_1/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
j
model_1/rangeRangemodel_1/range/startmodel_1/Rankmodel_1/range/delta*
_output_shapes
:
c
model_1/SumSummodel_1/Squaremodel_1/range*
T0*
	keep_dims( *
_output_shapes
: 
N
model_1/Rank_1Rankmodel_1/Placeholder_1*
T0*
_output_shapes
: 
W
model_1/range_1/startConst*
dtype0*
value	B : *
_output_shapes
: 
W
model_1/range_1/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
r
model_1/range_1Rangemodel_1/range_1/startmodel_1/Rank_1model_1/range_1/delta*
_output_shapes
:
n
model_1/MeanMeanmodel_1/Placeholder_1model_1/range_1*
T0*
	keep_dims( *
_output_shapes
: 
c
model_1/Sub_1Submodel_1/Placeholder_1model_1/Mean*
T0*
_output_shapes
:	�
S
model_1/Square_1Squaremodel_1/Sub_1*
T0*
_output_shapes
:	�
I
model_1/Rank_2Rankmodel_1/Square_1*
T0*
_output_shapes
: 
W
model_1/range_2/startConst*
dtype0*
value	B : *
_output_shapes
: 
W
model_1/range_2/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
r
model_1/range_2Rangemodel_1/range_2/startmodel_1/Rank_2model_1/range_2/delta*
_output_shapes
:
i
model_1/Sum_1Summodel_1/Square_1model_1/range_2*
T0*
	keep_dims( *
_output_shapes
: 
O
model_1/DivDivmodel_1/Summodel_1/Sum_1*
T0*
_output_shapes
: 
T
model_1/Sub_2/xConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
S
model_1/Sub_2Submodel_1/Sub_2/xmodel_1/Div*
T0*
_output_shapes
: 
h
model_1/Sub_3Submodel_1/Placeholder_1model_1/Reshape_1*
T0*
_output_shapes
:	�
S
model_1/Square_2Squaremodel_1/Sub_3*
T0*
_output_shapes
:	�
I
model_1/Rank_3Rankmodel_1/Square_2*
T0*
_output_shapes
: 
W
model_1/range_3/startConst*
dtype0*
value	B : *
_output_shapes
: 
W
model_1/range_3/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
r
model_1/range_3Rangemodel_1/range_3/startmodel_1/Rank_3model_1/range_3/delta*
_output_shapes
:
k
model_1/Mean_1Meanmodel_1/Square_2model_1/range_3*
T0*
	keep_dims( *
_output_shapes
: 
x
model_1/zeros_6Const*
dtype0*'
valueB��*    *(
_output_shapes
:��
�
model_1/VariableVariable*
dtype0*
shape:��*
	container *
shared_name *(
_output_shapes
:��
�
model_1/Variable/AssignAssignmodel_1/Variablemodel_1/zeros_6*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking(*
T0*(
_output_shapes
:��
�
model_1/Variable/readIdentitymodel_1/Variable*#
_class
loc:@model_1/Variable*
T0*(
_output_shapes
:��
�
model_1/Assign/value/0Pack2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model_1/Assign/value/1Pack2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model_1/Assign/value/2Pack2model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/mul_2*$
_output_shapes
:��*
T0*
N
�
model_1/Assign/valuePackmodel_1/Assign/value/0model_1/Assign/value/1model_1/Assign/value/2*(
_output_shapes
:��*
T0*
N
�
model_1/AssignAssignmodel_1/Variablemodel_1/Assign/value*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking( *
T0*(
_output_shapes
:��"	ԥ��      ���	�p���A*�3

mean squared error� D=

	r-squared  ��
�2
states*�2	   ���   @'�?    ��NA!َ�f`�r�)�x�x��@2�������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;��so쩾4�6NK��2�4�e|�Z#���-�z�!�p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�������:�              =@     @Q@     �e@     �v@     ��@     $�@     �@     �@     �@     4�@     ��@    ���@     ��@    ��@     �@     W�@     ��@    ��@    ���@    @��@    ���@    �y�@    �M�@    @}�@    ��@     �@     Y�@     ��@     ��@     ��@     ��@     ��@    ��@     ��@    ��@     ��@    ���@     p�@    �N�@     ��@     r�@     $�@    �v�@     ��@     ��@    ���@    �$�@     Y�@     ο@     F�@     3�@     ��@     ��@     �@    ���@    @��@    ���@    @��@    �N�@    @��@    ���@    @��@    ���@    0%�@    �;�@    �(�@    @f�@    �Z�@    �j�@    ��@     ��@     ��@    @u�@     Q�@    ���@     �@     �@     Y�@     /�@    �j�@    �e�@    @��@    ���@     �@    ���@    ��@    �=�@     ��@     ��@    �-�@    ��@     ��@     ��@     c�@     D�@     �@     K�@     �@     �@     ��@     ��@     �@     ��@     �@     f�@     ��@     l�@     D�@     X�@     ț@     Ț@     ̗@     `�@     T�@     đ@     ��@     ��@      �@     ��@     @�@     (�@     ��@     ��@      ~@     �}@     py@     �v@     pr@      q@     �n@     �k@     �n@     �g@     �c@     `b@     `b@      `@      [@      _@     �V@     �R@     �S@     @S@      N@     �K@     �I@     �L@      E@      =@      ;@      A@      @@      =@      8@      =@      0@      0@      *@      (@      3@      *@      *@      ,@      $@      *@       @      @      @      @      @      @      &@      "@      @      @      �?      �?       @      @      @      @      @      �?      �?      �?      �?       @       @      @              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?               @       @      �?              �?      �?       @      @       @       @      �?              @      @      @      @      @      @      @      (@      @      @      "@      0@      .@      &@      ,@      0@      &@      4@      7@      7@      6@     �@@      3@      ?@      E@     �G@     �B@     �C@      E@     �L@      J@      T@      Q@     @P@     �U@     @T@     @Z@     �\@     �[@     �a@      `@     �e@     �g@     �f@     �k@     �j@     Pp@      q@     �t@      v@      x@     px@     �y@     0}@     �@     ��@      �@     ��@     ��@     ��@     H�@     ��@     ��@     �@     d�@     �@     �@      �@     ԛ@     М@     d�@     ҡ@     ��@     4�@     &�@     Ʃ@     ƫ@     ��@     �@     �@     ��@     ��@     w�@     ֹ@     C�@     ��@     ��@    ���@     ��@     *�@    ���@     ��@    @s�@    �P�@    @9�@    @��@    ���@    ���@     ��@    ���@     ��@    ��@     ��@    ���@    @��@    @��@    �:�@    �3�@     !�@     ��@    �x�@    ���@     ��@    `0�@    @��@    �G�@    �:�@    ���@    @��@    �r�@    ���@     ��@    �6�@    ���@     ��@    �w�@     ��@     ��@    ��@    ���@    ��@    ���@    ���@     ��@     ��@    ���@     B�@     o�@     8�@    �2�@    ���@     W�@    �%�@     ��@    @i�@    ��@     ��@     !�@    �`�@    ���@    ��@    @��@     '�@    @��@     ��@     W�@     �@    ���@    ��@     ��@     ��@    ���@    ��@    ���@     ��@     ��@     I�@     x�@     ��@     d�@     X�@     ��@     �r@     �`@     �D@      @        
�
predictions*�	    Q`?   @��b?     ί@!  F�cg!@)�<zx#�?2(E��{��^?�l�P�`?���%��b?5Ucv0ed?�������:(              a@     ��@      "@        �JT�      6Q��	<K����A*�5

mean squared error1D=

	r-squared ���
�3
states*�3	   `
j�   �Pf�?    ��NA!,$�G_��)6kX��r�@2��P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4�6NK��2�7'_��+/��'v�V,�4��evk'���o�kJ%�4�e|�Z#���R����2!K�R���9�e��=����%�=�mm7&c>y�+pm>�z��6>u 5�9>�`�}6D>��Ő�;F>��8"uH>6��>?�J>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�������:�               @      @      ?@      \@     �j@     �@     Ԑ@     ��@     ަ@     n�@     ��@     ֹ@     ѿ@     ��@    �d�@    ���@    ���@     ��@     }�@     ��@    ���@     ��@    ��@    @�@    ���@    ���@    ���@    ���@    @K�@     ��@    @��@     U�@     ��@     B�@    @ �@     ��@     w�@    ���@     ��@     V�@    �e�@     m�@     ��@    �)�@     ��@     w�@     ��@    ���@    ���@    ���@    @�@    �X�@    ���@     ��@    @9�@     `�@    ���@     ��@    �H�@    �4�@    �(�@    ��@    �x�@    `)�@    `��@    ���@    ���@    @��@    ��@    �\�@    `e�@     �@    ��@    �g�@     ;�@     y�@     ��@    `U�@     ��@    ���@    @��@    @��@     J�@    ���@    �$�@    ���@    �&�@     ��@     ��@     ��@     ��@     R�@     ��@     ~�@     �@     p�@     �@     ��@     w�@     ��@     K�@     ��@     ��@     �@     ��@     .�@     ��@     �@     �@     d�@     Ȥ@     B�@     H�@     ��@     \�@     ؙ@     ��@     d�@     \�@     0�@     ��@     ȍ@     ؋@      �@     p�@     ��@     ��@     ��@     P|@     �}@     @x@     v@     �s@     �t@     �r@     �l@     `j@     �h@     �g@     �b@     �`@      d@     @^@      [@     �Y@     �Y@      S@      R@     @S@      H@      Q@      M@      F@      D@     �E@     �B@      ?@      :@      ;@      8@      3@      3@      1@      4@      1@      $@      ,@      &@       @      &@      &@       @      @      @      @       @      @      @       @       @       @      @      @      @      �?      @               @      �?      @       @      �?      @               @              �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?       @       @               @       @       @      @              @       @      @      @      @       @       @      @      @      @      @       @      "@      "@      @      0@      *@      3@      (@      *@      1@      $@      =@      9@      ;@      7@      <@      2@      9@     �D@      H@      B@      K@      M@     �M@     �P@     @V@     @T@     �T@     @V@     @_@     �^@     �^@     `a@     @e@     �f@     `g@      k@      m@     `m@     `o@     Ps@     �t@     py@     @x@     �x@      }@     `{@     p�@     ��@     ��@     ��@     8�@     h�@     @�@     H�@     ؑ@     ̒@     ��@     ��@     l�@     t�@     X�@     L�@     ��@     �@     ��@     l�@     t�@     ~�@     Ю@     �@     %�@     �@     +�@     �@     ��@     S�@     ��@     ��@     ��@     ��@     5�@    ���@    �Y�@     ^�@     ��@    ���@    ���@     ��@     ��@    ���@    @��@    @��@     _�@    @K�@    @`�@    �w�@    ���@    ���@     D�@    �h�@     ��@     !�@    `I�@    @��@    @��@    �1�@     -�@    @��@    ���@    ���@    @��@    �e�@     �@    �8�@    ���@    �f�@     D�@    @)�@    ���@    �*�@     ��@     k�@     ��@    ���@     ��@     ��@     ��@     �@    ���@     =�@    ���@    ��@     ��@    ��@     ��@     ��@     ��@    �<�@     ��@    ���@    �f�@     U�@    ���@    ��@    @K�@     �@    @>�@     ��@    �L�@    ���@     ��@    �j�@    @9�@    @�@     ��@    �`�@     ��@     ��@    �
�@     =�@     ��@     �@     ,�@     �@     �@     ��@     ��@     Ё@     �s@     @d@     �I@      &@        
�
predictions*�	   �� s?    �Ux?     ί@!  ��r5@)s���/��?20uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:0             �J@      �@     ��@      �?        }�x�      7=�]	��&����A*�5

mean squared error�6D=

	r-squared ��
�3
states*�3	    ���    ���?    ��NA!�<8dud��)`�}H� �@2��E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH��`�}6D�/�p`B�����W_>�p
T~�;���o�kJ%>4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>u 5�9>p
T~�;>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�������:�              @      A@     ``@     �z@     ��@     ܘ@     �@     ��@     �@     v�@     e�@     w�@      �@     �@    ���@     L�@    ���@    ��@    ���@     ��@    ��@     ��@     ��@    @��@    ��@    @��@     o�@     ��@    ���@    �Z�@    @�@    @/�@    ���@    ��@     "�@     F�@     ��@    �N�@    ��@    ���@    �~�@     ��@    ��@    �u�@    ���@    �$�@    @��@    �y�@    �1�@     ��@    @��@     ��@     ��@    ���@    `/�@    `��@    @��@    @C�@     d�@    ���@    ���@    ��@    ��@    �+�@    @��@    @��@     ��@     �@    �b�@     ��@    ��@    ��@    ���@    ���@     ��@    ���@    ��@    ���@     ��@    �L�@    �v�@     ��@    �|�@     ��@     O�@     N�@    ���@     ��@     ��@    ���@     ��@     J�@     ��@     ]�@     �@     i�@     ٳ@     Z�@     ��@     �@     .�@     ��@     �@     ��@     *�@     
�@     ��@     �@     X�@     0�@      �@     (�@     h�@     h�@     ��@     `�@     �@     ��@     ��@     h�@     ��@     �}@     �}@     �x@      {@     @w@     �s@     Pr@     Pq@     p@      j@     �m@     �g@     `e@      d@     `b@     @b@     �\@     �]@      [@     @W@     @V@     �R@     �P@     �N@     �O@      M@      B@     �H@     �D@     �B@      @@     �B@      <@      7@      4@      ;@      7@      8@      0@      ,@      &@      &@       @      $@      .@      "@      "@      @      @      @       @      @      @      @      @      @       @       @      @      �?      @      @              �?       @       @       @       @      �?       @      �?       @       @              �?               @              �?              �?      �?      �?              �?      �?       @              �?              �?              �?      @               @              �?      �?              @       @      �?      @       @      �?              @      @      �?      @      @       @      �?      @      @      @      @      @      @      @       @      @      &@      @      $@      &@      ,@      0@      *@      4@      6@      =@      :@      6@      A@     �A@     �@@      A@     �G@      N@     �G@     �J@     �N@     �U@      R@     �S@     �X@     �[@      X@      `@     �d@     �d@     @c@     �i@     �i@      m@      o@     �r@     �s@     �t@     @w@     �w@     �{@     �y@     ��@     ��@     h�@     ��@     8�@     0�@     Ȋ@     ��@     @�@     �@     D�@     ��@     ��@     �@     \�@     h�@     ��@     ��@     ��@     �@     ��@     ڦ@     T�@     ��@     ��@     ɰ@     k�@     =�@     c�@     @�@     1�@    ��@      �@    ���@     +�@     ��@    ���@    �\�@    ���@    �z�@    �z�@    ���@    @��@    �+�@    @&�@     ��@     �@    ��@     ��@    @��@    `;�@    `��@    ���@     ��@    `��@    ���@     �@    �_�@    �+�@    �2�@    �-�@    @�@    ��@     +�@    @��@    �s�@    �$�@     %�@    @8�@     ��@    ���@    ���@    �,�@    �J�@    ���@     )�@    ��@    �E�@     ��@    �-�@     i�@     ��@     T�@    �c�@    ���@     y�@     ��@     R�@     l�@    ���@    @�@     �@    ���@    @"�@    @(�@    @��@     -�@    @�@    �l�@     5�@    �2�@    ���@     �@     ��@    �+�@    �>�@     ��@    ���@    �p�@    ��@     ��@     �@     �@     �@     �@     �@     ��@     ��@     ��@     0�@      n@     @P@      "@      �?        
�
predictions*�	   �}s?   �!)|?     ί@!  N��t7@)m�%Y�?28uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:8              @     `s@     ,�@     ��@     �I@        ���      �_��	�.����A*�4

mean squared error�&D=

	r-squared ��
�2
states*�2	   ��o��   @��?    ��NA!O��S���)4��91�@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c���u}��\�4�j�6Z�Fixі�W��'v�V,����<�)��i
�k���f��p���f��p>�i
�k>�'v�V,>7'_��+/>�so쩾4>�z��6>�`�}6D>��Ő�;F>6��>?�J>������M>�
L�v�Q>H��'ϱS>��x��U>��u}��\>d�V�_>w&���qa>�����0c>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              @      0@     �G@     @a@     0s@     Ȁ@      �@     Ж@     ��@     |�@     ˰@     9�@     �@     �@    ���@    �"�@    ���@      �@    ���@     ;�@     [�@    �E�@    @�@     ��@     ��@    @F�@    ���@     ��@    @��@     R�@    @��@    @��@    �n�@    ���@    ���@    @x�@     ��@    ���@    ���@     ]�@    ���@    ���@    ���@    ���@    @M�@    ���@     ��@    @x�@    ���@     =�@    @b�@     r�@     }�@    �)�@     ��@    �
�@    �U�@     ��@    ���@    �y�@     
�@    ���@    ���@    �B�@    @��@    �^�@    ���@    �@�@    `��@     J�@    �f�@    �X�@    @��@    ���@    @%�@    ���@    �"�@    �8�@    ���@    ���@     ��@    �p�@    �N�@     ��@     ��@     ��@    �a�@    ���@    ���@     q�@     4�@     E�@     �@     ��@     2�@     ��@     ��@     B�@     ��@     ��@     Z�@     ʦ@     ��@     ¢@     �@     x�@     (�@     ��@     \�@     @�@     ��@     ��@     @�@     ��@     ��@     �@     (�@      �@     ��@     X�@     �}@     �{@     @v@     �w@      u@      s@     �p@     �o@     �k@     �g@      g@     �d@     �d@     �b@      ^@     @\@     �[@     @W@     @T@     @S@     @R@      P@     �H@     �G@     �G@     �A@      E@      C@      D@      <@      ?@      7@      4@      7@      .@      =@      5@      .@      1@      "@      0@       @      "@      "@       @      @      @      @      $@       @       @      @       @      @      @      �?      �?      �?       @       @      �?      @      �?       @      �?      �?      �?               @      �?               @               @              �?              �?              �?               @              �?              �?      �?              �?       @      �?              @      �?       @      @      �?      �?              @      @      @      @       @      @      @      "@      @      @      "@      @       @      $@      ,@      @      $@      (@      0@      5@      3@      5@      0@      3@      <@      =@      >@     �B@      C@     �B@      J@      G@      E@      L@     @R@      L@      U@      Y@     �[@      R@     �X@     �]@     �`@     �a@      d@      g@     �g@      k@     �k@     0p@     �q@     @q@     Pu@     pv@      z@      |@     0�@     8�@     H�@     `�@     ��@     Ȉ@     ��@     ��@     4�@     4�@     @�@     $�@     ��@     �@     H�@     ��@     F�@     ��@     .�@     (�@     �@     ��@     ��@     ��@     ��@     �@     ɴ@     Ե@     ָ@     Z�@     ��@     ��@     ��@    �E�@     ��@    ���@    ���@    ���@     *�@    ��@    ���@    ���@    @��@    �`�@    ���@    ��@    ���@    ���@     �@    ���@    �p�@    �t�@     ��@    �r�@     y�@    ���@    `(�@    `��@     ��@     P�@     ��@    ���@    �j�@    �/�@    �q�@    ���@    @��@    ���@    @}�@    �O�@    �G�@    ��@     ��@    �"�@    ���@     ]�@    �g�@    ���@    @��@     W�@    ���@     ��@     ��@    ���@    @��@     ��@    @h�@     ��@    @��@    @��@     �@    ���@     ��@    �2�@    @\�@    @1�@    @)�@    ���@    @��@     w�@     �@    ���@     U�@     &�@    ���@     �@    �;�@    ��@     ��@     �@     ��@     �@     D�@     $�@     �@     `}@     �j@     �S@      7@      �?        
�
predictions*�	    6.f?    q!�?     ί@!  ؜��3@)L�ɓ@@�?2x5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?�������:x              �?      @      3@     ``@     �y@      �@     p�@      �@     �@     �r@      ^@      8@      �?        �
;B"      �:V	ѐހ���A*�D

mean squared error#UC=

	r-squared�/�;
�1
states*�1	   ��    S�@    ��NA!G�@Ů�)b��	��@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol������0c�w&���qa�d�V�_���u}��\�H��'ϱS��
L�v�Q�28���FP�������M��`�}6D�/�p`B���8"uH>6��>?�J>28���FP>�
L�v�Q>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>BvŐ�r>�H5�8�t>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?       @       @      @      1@      @@     �X@     `m@     �}@     ��@     �@     �@     �@     2�@     �@     �@     ��@    ��@     ��@    ���@    ���@    ���@    � �@     �@     ��@    @"�@    �5�@    ���@    �&�@    @��@     ��@    �z�@     ;�@    ��@     ��@    �&�@    @<�@    ��@    �5�@    �x�@    @��@    �%�@    `��@    ���@    `<�@     ��@     ��@     ��@     <�@    ���@    @��@    `C�@    `��@    @��@    @��@    @��@    `��@    `��@    `�@    �'�@     ��@    ���@     ��@    @_�@    ���@    `��@     �@    ���@    �K�@    �h�@     ��@     -�@    ���@    @�@    @��@     d�@     ��@     ��@     ��@     ��@     ��@    ���@     �@    � �@     ��@     �@     �@     м@     �@     �@     ǳ@     Z�@     d�@     �@     ��@     �@     ��@     ��@     ܡ@     ��@     ��@     ��@     d�@     (�@     t�@     ܐ@     `�@     X�@     Њ@     �@     �@     ��@     ��@     h�@     �~@     �{@     �x@     pv@     �u@     �t@     �p@     p@      m@     �g@     @h@     �f@      c@     �c@      c@     @\@     �Z@      X@     @U@     �U@     @U@     @T@     �P@     �E@     �F@     �F@      J@     �D@      F@      @@      7@      5@      6@      8@      9@      9@      3@      &@      ,@      $@      "@      "@      &@      ,@      @      @      "@      @       @      @      @      @      @      "@      @              @              �?       @      @      @              �?       @              �?      �?      @       @              �?               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?      @      �?      �?      �?      @       @      @       @      @       @      @      @      @      @      @       @      @       @      @      (@      "@      $@      (@      0@      3@       @      1@      5@      7@      =@      9@     �A@      ;@      <@     �F@      F@     �I@      K@      H@     �I@     @U@     �U@     @W@      Q@      ]@     �Z@      a@     `a@     �e@     �f@     �d@     �l@     �l@     0q@     �q@     0r@     Pu@     0u@     py@     �y@     �}@     ��@     `�@     �@     ��@     (�@     ؉@     X�@     H�@     ��@     �@     �@     d�@     ��@     М@     Ơ@     B�@     �@     x�@     �@     ��@     ��@     �@     Z�@     ��@     ��@     з@     z�@     ��@     ��@    �(�@     ��@     H�@    �u�@     l�@    ���@     l�@    @��@    ���@     ��@    �0�@    ���@    �M�@    @��@     r�@     ��@    @�@     	�@     �@    @�@     ��@     ��@      �@     [�@    `U�@    �x�@    ���@     ��@    @��@    @��@     x�@    �6�@    �"�@    ��@    ���@    @��@    ��@    ���@    `u�@    �k�@    �^�@    �6�@    @R�@     �@    ��@     \�@    ���@    ���@    ���@     ��@     x�@    ���@     ��@    @�@    �3�@    ���@     ��@     ��@    @w�@    @��@    ���@    �"�@    @��@    ���@    ���@     ��@     u�@     ^�@    ��@     p�@     ͼ@     =�@     ��@     ޤ@     ș@     ��@     8�@     @p@     @Z@      K@      (@      (@      @      �?        
�
predictions*�	   �b���   `��?     ί@! �*� C@)�_3��m�?2�	��<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7���6�]���1��a˲�pz�w�7��})�l a�5�"�g���0�6�/n���u`P+d����n������ߊ4F��>})�l a�>>�?�s��>�FF�G ?����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�	              �?      �?               @      @      @      @       @      &@      @      "@      @      1@      "@      2@       @      &@      7@      7@      5@      ;@      4@      4@      7@      9@      ,@      :@      .@      2@      *@      &@      *@      ,@      @      &@      (@      "@      "@      @      @      @      �?      @      "@      @      @      @       @      @      @      @      �?      �?      �?               @               @              �?               @              �?              �?              �?              �?              �?              �?              �?              @      @      �?      �?      @      @      @      @      @      �?       @      @      @      @       @       @      @      @      "@       @      @      "@      $@      &@      (@      ,@      $@      0@      8@      2@      0@      4@      8@      ;@      :@     �@@      J@     �H@      I@     �H@      O@     @P@     @U@      \@     @Z@     �_@      Y@     ``@     �c@     `f@     �g@      e@      g@      e@     �e@     �c@     `c@     `c@     ``@     �W@     �[@     �Q@      C@      :@      0@      @      @      �?        ���f""      ��d�	��;����A*�D

mean squared error�0D=

	r-squared ���
�.
states*�.	   �9���    L� @    ��NA!�Fu�˵��)�^�r A2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol������0c�w&���qa�4��evk'>���<�)>�z��6>u 5�9>��u}��\>d�V�_>=�.^ol>w`f���n>E'�/��x>f^��`{>[#=�؏�>K���7�>u��6
�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�               @      ,@      B@     �Z@     q@     ��@     ,�@     ��@     e�@     ˼@    ���@     6�@     {�@     ��@    �!�@    ���@    @@�@     ��@    ��@    @J�@    @g�@    �V�@    ���@    @��@    @`�@    `�@    `;�@    @�@     ��@    �%�@    @�@     ��@    @!�@    `��@    `p�@    ���@    ��@    ���@    � �@    �@�@     ��@     ��@    `��@    @w�@    ���@    `��@    `�@     t�@    `��@    `��@     ��@     ��@    @��@    ���@    �9�@    �(�@    ��@     E�@    @��@     ��@    ��@     ��@     '�@     ��@    ���@    ���@     j�@     ��@     ��@     �@    �o�@    �$�@     ٽ@     ��@     �@     e�@     ��@     /�@     ��@     �@     6�@     V�@     ��@     Ц@     ��@     Ƣ@     �@     ��@     4�@     L�@     �@     ��@     4�@     ��@     ؐ@     ��@     ��@     �@     (�@     Є@     X�@     (�@     ��@     P|@     0z@     @v@     �r@     �t@     �r@     �p@     �o@     @h@      g@      d@     @c@     �b@     �]@     �a@     �V@     @]@     @S@     @S@      P@      N@     �J@      P@     �J@      J@     �C@     �B@      ?@      @@     �@@      >@      9@      5@      5@      4@      $@      2@      &@      .@      "@      @      @      "@      @      @       @      @       @       @      @              @      @      @       @      @               @      @      �?              �?      �?      �?      �?      �?              �?              �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @      �?      @      @      @      �?              @      @      @       @      @      @      @      @       @      @      @       @       @      *@       @      .@      *@      *@      0@      &@      3@      1@      1@      .@      A@      :@      6@      =@      F@     �E@      C@     �I@      K@     �I@      O@     @U@      X@     @V@      \@     �W@     �`@     @^@     �b@     `f@     `e@     �f@     �i@      m@      q@     q@     �r@     �u@      v@     �y@     �{@     ��@     h�@     ��@     0�@     ��@      �@     Ȍ@     �@     ��@     L�@     ܓ@     ԕ@     ��@     �@     ��@     ̟@     6�@     Ȣ@     ~�@     h�@     ̩@     ��@     ̮@     K�@     ��@     @�@     J�@     �@     ��@     ��@    ��@    ���@     ��@    ���@     O�@    ���@    �t�@     b�@    ���@    @{�@    @��@    @F�@    ���@    ���@    ���@    �L�@    @��@    ���@    ���@    ��@     ��@    @��@    ���@    ���@    ���@     ��@    `>�@    ���@     A�@    ���@    `��@    ��@     l�@    @�@    ���@    ���@    ��@    �h�@     ��@     A�@    ��@    �=�@    �z�@    ���@     ��@    ��@     @�@    �M�@    ���@    �R�@    ���@     ��@    ���@    �S�@     ��@     ��@    �O�@     �@     T�@     ��@    �p�@    ���@    ���@     ��@     �@     ��@     ��@     ��@      j@     �L@     �@@      &@      @       @        
�
predictions*�	    @忿    o�?     ί@!  X�/Y&@)�`�c1�@2�
Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9��x?�x��>h�'��f�ʜ�7
�������iD*L�پ�_�T�l׾
�/eq
�>;�"�q�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�
              �?      @       @      *@      5@      A@      H@     �K@      Q@      T@     �V@     �W@     @W@      X@     �S@      X@     @W@     �W@     �T@     �U@      R@     �N@      M@     �L@     �C@      B@     �D@      C@     �F@      A@     �A@      ;@      =@      4@      6@      .@      (@      .@      2@      ,@      *@      (@      "@      "@      "@      @      �?      @      @      @      @      @      @      @      �?      @      �?      @       @       @      @      �?       @       @      �?      �?              �?      �?      �?              @              �?              �?               @              �?              �?      �?      �?              �?               @              �?              �?      �?      �?      �?      @               @      @      �?      @      �?      @      �?      @      @      @      @      @      @      @      @      "@      &@      "@      1@       @      "@       @      0@      .@      ,@      3@      6@      :@      <@      :@      8@     �B@      G@     �D@     �I@      M@      D@     �I@     �F@     �O@     �L@     @R@     �Q@     @S@      Q@     @S@      S@      W@     �W@      S@     @Q@     �R@     �R@     �N@     �Q@     �D@      7@      C@      1@      (@      &@      @              �?      �?        +�}��"      �x��	������A*�D

mean squared error�A=

	r-squared@iO<
�.
states*�.	   �p���   @h� @    ��NA!�i��w��)�F=�XA2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
�������~�f^��`{�E'�/��x��H5�8�t�BvŐ�r�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�Z�TA[�>�#���j>%���>��-�z�!>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �?      �?      0@     @[@     ��@     l�@     �@     �@     ��@    ��@     �@     b�@     K�@    @.�@    @4�@    ���@    ���@     ��@    ���@     ��@    ���@     ��@    ���@    ��@     ��@    ���@    @��@    @��@    �~�@    @��@     ��@    ��@    @U�@     ��@    ���@    ���@    � �@    �/�@    `L�@    �<�@    ��@    �~�@    @��@    ���@    `]�@     ��@    `��@    `�@    ���@    `��@    @��@    ���@     ��@    ��@    �4�@    ���@    �3�@     ��@    ��@    ���@    @z�@    �C�@    ���@     ��@    ���@    @o�@    ���@    ���@    @��@     �@    ��@     [�@     ��@    �!�@     !�@     ��@     %�@     ��@     W�@     ��@     8�@     ��@     �@     ɲ@     %�@     �@     ޫ@     ��@     t�@     z�@     0�@     F�@     ��@     �@     t�@     (�@     <�@     ܓ@     (�@     ��@     ��@     ��@     �@     ��@      �@     Ђ@     x�@     `}@     �{@     �x@     `u@     0v@     �u@     @q@     �n@      j@      i@     �e@     `f@      f@     �`@     ``@      X@      ]@     �[@     �U@     �S@      I@     �Q@     �M@     �L@      M@      D@      F@      >@      5@      ;@      8@     �@@      4@      4@      3@      *@      "@      @      1@      ,@       @      $@      @      @      @      &@       @      @       @      @       @      @      @      @      �?       @      @               @       @              �?      �?      @      @              @      �?               @              �?      �?      �?              �?              �?              �?      �?      �?              �?       @              @       @              �?      �?      �?       @              @      �?       @      @      @      @              �?      �?      @      @       @      @      @      @       @       @      "@      @      "@       @      ,@      ,@      0@      ;@      $@      0@      8@      5@      ;@      >@      E@      @@     �G@     �F@      K@     �J@     �L@      P@     @S@      X@     �U@     �Z@     @Z@     @X@     @^@     �a@      g@     �c@     @i@     �h@     �j@     �p@     Pq@     Pt@     �s@     0w@     �z@     �{@     p@      �@     x�@     ȃ@     ��@     (�@     ؊@     ��@     ؏@     t�@     x�@     ��@     |�@     ��@     ��@     d�@     ��@     *�@     ��@     >�@     ��@     �@     "�@     C�@     ��@     <�@     @�@     ٷ@     ׺@     q�@     �@    �%�@    ���@     ��@     �@     ��@     =�@     ��@    ���@     ��@    �$�@    ��@     K�@    @��@     ��@     :�@    ���@     y�@     ��@     >�@     �@     ��@    �R�@    ���@    @3�@    ���@    �m�@    ���@     ��@    ���@    �h�@     M�@    ��@    `��@     ��@    @��@    @J�@    ��@    @	�@     ��@    @��@    ���@    ���@    �9�@    ���@     ��@     z�@    @�@     �@    �w�@     ��@    ��@    @+�@    ���@     e�@     ��@     ��@    ���@     ��@     ��@    ���@    ���@    @��@    @��@    @��@    ���@    �V�@     ��@     ��@     �@    ���@     M�@     ��@     P{@     �b@     �L@      "@      @      �?        
�
predictions*�	   @�飿   `���?     ί@!  �]��)@)��Mbt�?2�
`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[����Zr[v��I��P=��a�Ϭ(���(��澢f�����uE���⾮��%�jqs&\�ѾK+�E��Ͼ��(���>a�Ϭ(�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�
               @      "@      "@      (@      4@     �@@      K@     @P@     �P@     �P@     @U@     �S@      U@      T@     �S@      T@     �W@     @R@     @P@     �G@     �N@      E@     �E@      F@      G@     �B@     �C@      E@     �B@      5@      8@      7@      7@      9@      3@      *@      &@      "@      @      "@      &@      $@      @       @      @      @      @      @               @      @      �?               @      @      �?      @      @      �?              �?      �?               @       @       @      �?      �?       @      �?       @              �?              �?              �?      �?              �?               @              �?              �?              �?      �?      �?       @      @              @      @       @      �?               @      @              �?      �?       @              @      �?       @      �?       @      @      $@      �?      @      @      "@      @      &@      $@      &@      "@      &@      "@       @      .@      *@      9@      2@      2@      7@      9@      A@      :@     �D@      B@     �D@     �F@     �I@      H@     �J@     �L@     @Q@     �S@     �T@     @T@     �R@      V@     �U@     �Y@     �U@     @U@     �S@     �W@     �S@     �R@     �T@      R@     �I@     @R@     �D@     �B@      3@      2@      .@      @      @      �?        ���!      Չ�~	�D����A*�B

mean squared error@=

	r-squared�ǩ<
�-
states*�-	   @�n��   �r��?    ��NA!6��`#��@)�4\td
A2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�������m!#���
�%W��T�L<��u��6
��[#=�؏�������~��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol���u}��\�4�j�6Z��i����v>E'�/��x>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @      F@     @e@     �u@     P�@     ��@     \�@    �L�@     G�@    �!�@    ��@     ��@     ��@    ���@    ��@     ��@    ���@    ���@     �@    � �@    ���@    ���@     ��@     �@     b�@    `��@     %�@     \�@     ��@    �F�@    `��@    ���@    ���@    �N�@     ��@    ��@    ���@     ��@     ��@    ��@    � �@    �$�@     �@    `��@     t�@    �`�@     �@    @��@    �D�@     ��@    �"�@    �_�@     ��@    @��@    ���@    @��@     !�@     j�@     �@    �o�@    @��@    �t�@    �O�@    ���@    ���@    ���@     ��@    �?�@     t�@     p�@    �m�@    �<�@     *�@     X�@     E�@     ��@     I�@     c�@     �@     $�@     J�@     ��@     ħ@     ��@     �@     �@     0�@     T�@     �@     ��@      �@     ��@     Ē@     �@     p�@     Њ@     8�@      �@     X�@     `�@     0�@     �~@     ~@     �z@     �w@     `v@     �r@     Pp@     �p@     `m@     �h@      h@      i@     �c@      b@     �`@     �\@      Y@     @Z@     �Y@     �V@      U@     �R@     @Q@     �I@     �I@     �E@      K@      E@     �D@      D@      @@      8@      2@      3@      5@      6@      1@      0@      "@      @      "@      @      &@      @       @       @      @      @      @      @      @       @      @      @      @       @      �?       @              @      @       @       @      �?              �?               @              �?              �?              �?       @               @              �?              �?              �?      �?      �?      @      �?       @      @       @      @      @      @      �?      @      @      @       @      @      @      @      @      @      @      @      @      @      @      @      &@      ,@      (@      @      *@      &@      *@      1@      5@      1@      7@      3@      ?@     �@@     �B@      ?@      A@     �E@      O@      K@      N@      P@      Q@     �S@     �T@      [@      ^@      ^@     @`@      b@     �b@     �d@     �f@     �k@     �m@      p@     �r@      r@      v@     �w@     �v@     �z@     @�@     �@     �@     P�@     ��@     ��@     0�@     ��@     ��@     ��@     (�@     ��@     @�@     d�@     М@     ��@     �@     �@     
�@     V�@     ��@      �@     p�@     ˱@     ��@     K�@     �@     �@     �@     )�@    ���@     �@     ��@    ��@    ���@     ��@     �@     L�@    ���@    @��@    @i�@    ��@    �<�@    �
�@     ��@    ��@     �@    ���@     V�@    ���@    ���@    �^�@     ��@    �Z�@    ���@    �&�@    ���@    ���@    `�@    ��@    `H�@     a�@    �>�@    �z�@     Y�@     e�@     J�@    ��@    �3�@    @��@    `��@    ��@     >�@    ��@    ���@    ��@     7�@     ��@    �J�@     ��@    @L�@    ���@    �^�@    �2�@    �\�@    ���@    �=�@    ���@    ���@    ���@     �@    �N�@    �N�@     {�@    �i�@     �@      �@     P�@     `l@      X@     �A@      $@      �?        
�
predictions*�	    +���   �h��?     ί@! ����E@)��TL�	@2�
�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲�8K�ߝ�a�Ϭ(�;�"�qʾ
�/eq
Ⱦ>�?�s��>�FF�G ?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:�
              �?               @      .@      5@     �B@      O@     �R@     �V@     @\@     �S@      T@     �S@     @T@     �U@      S@      M@     �N@     �E@      ?@      D@      D@      G@      A@     �@@      @@      ?@      >@      2@      1@      *@      ,@      0@      1@      1@      @      @      @      .@      @      @       @      @      @       @      @      @      �?      �?       @      @      @       @      @      @      @      @              �?               @              �?       @              �?      �?              @              �?               @              �?               @              �?              �?              �?              �?               @              �?              �?              �?      �?       @       @              @              @      @      @      @      @      @      @      @      @      .@      @      (@      (@      (@      ,@      (@      4@      7@      9@      .@      8@      2@      9@      ;@      >@     �B@     �E@      F@     �D@     �J@     �C@      N@     �N@      J@     �O@     �R@      S@     @T@     @S@     @V@      U@      W@      U@     @U@      V@     �X@      X@     �Y@      V@     �V@     �T@     �O@      H@      D@      5@      7@      0@      "@      @        V�pP�!      ��F�	bIM����A*�C

mean squared error�#>=

	r-squared���<
�-
states*�-	   ��<�    q��?    ��NA!=�Ҋ>̺@)�T�-��A2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7���i����v��H5�8�t�ڿ�ɓ�i�:�AC)8g�28���FP�������M�Fixі�W>4�j�6Z>d�V�_>w&���qa>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>�H5�8�t>�i����v>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �?      *@     @W@     �c@     @k@     �r@     �x@     ��@     ��@     A�@     �@    ���@    �f�@    @}�@    ���@    ���@    ��@    @K�@    @D�@     ��@    @��@     ��@    @��@    ���@    @��@    @K�@     ��@    `��@     B�@    `��@    �j�@    �q�@    �6�@    �:�@    ��@    `��@    @p�@    `��@    ���@     ^�@    �=�@     �@    @��@     ��@    @��@    ��@    �4�@    �x�@    �y�@     ��@    ���@    ���@    �~�@     ��@    ���@    ��@     �@    �h�@    ���@    �H�@    @��@    �=�@    ���@     ��@    �P�@    �l�@     ��@     ��@     �@     "�@    ��@     ��@     ,�@     4�@     ��@     ��@     ��@     G�@     b�@     l�@     ʪ@     .�@     ��@     ��@     R�@     ��@     l�@     ܜ@     d�@     �@     ��@     ��@     ȑ@     ��@     Ќ@     ��@     (�@     h�@     P�@      �@     x�@     P~@     P{@     �x@      v@     �s@     pq@     �q@     `l@     @l@      k@      e@     �e@      `@      `@     �`@     �Z@      ]@     �X@     �T@     @U@      N@     �P@     @Q@     �F@     �L@      G@      F@     �B@      >@      @@      >@      3@      ,@      5@      5@      1@      $@      .@      @       @      ,@      &@      $@      @       @      @       @      @      @      @      �?      @      �?      @       @      @       @      @      @      �?       @       @      �?       @      @       @       @              �?      �?               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?      �?              @       @       @      @      @      �?       @      @      @      @      @      "@      @      @      $@      &@      "@      &@      ,@      $@      ,@      4@      1@      1@      3@      3@      9@      5@     �@@     �B@      A@     �C@      C@     �L@     �G@      L@      N@     �R@      S@     @T@      W@     �Y@      W@     �\@     @]@      a@     @c@     @e@     �f@     �h@      i@     �l@     �n@     �s@     �s@     �w@     �x@     �z@     |@     �@     `�@     ��@      �@     ��@     ؊@      �@     ��@     Б@     X�@     |�@     ��@     �@     L�@     �@     ֠@     �@     |�@     �@     x�@     ,�@     l�@     ��@     �@     �@     ͵@     J�@     \�@     �@     ��@     :�@    ���@    ���@     !�@    �I�@    ���@    ���@     �@     ��@    �-�@    @d�@    ���@     ��@    @��@    @	�@    ��@    ��@    @��@    �/�@    �v�@    `b�@     O�@    ��@    `��@     ;�@    ���@     �@    �y�@    `��@    ���@    @��@    ���@    ���@     ��@    ���@    �k�@    ``�@    ���@    �C�@     ��@    ���@    �>�@     ��@     ��@    �5�@    @��@    @=�@    �Y�@    @a�@    ���@    ��@    ���@    @��@    ���@    ���@     ��@    �5�@    @}�@    ���@     ��@     ��@     )�@     �@     ��@     `m@     �X@      H@      "@      @        
�
predictions*�	    �J��   ���?     ί@!  �#��	@)CO�x�Z @2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F��K+�E���>jqs&\��>�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?�������:�
              �?      �?      @      (@      ;@     �S@     �\@     `a@     @d@     `c@     �c@      _@     �a@     �_@     @X@     �X@     �[@      U@     @P@     �P@     �H@     �I@      H@     �L@     �E@      :@      <@      8@      <@      =@      ;@      ;@      4@      5@      6@      2@      "@      (@       @      *@      &@      @      "@      @       @      @      �?      @       @      @       @      @      �?       @               @              @      �?      �?      �?      �?              �?               @      �?               @              �?              �?               @              �?              �?      �?      �?      �?       @              �?      �?      �?      �?              �?              �?      @       @      �?      �?       @      @       @      �?      @      @      @      @      @      @      @      @       @      @      (@      &@      @      0@      0@      4@      &@      *@      (@      8@      8@      5@      6@      >@      6@      >@      <@     �E@     �B@     �D@      B@     �C@     �E@     �B@      E@     �H@     �G@     @P@     �L@     @Q@     �K@      P@      Q@     �L@     �O@     �P@     �N@      M@      C@      B@      <@      =@      *@      &@       @      @      @      �?      �?        76��R"      ��aC	5������A	*�D

mean squared errorO�==

	r-squaredp=
�/
states*�/	    ��   ��= @    ��NA!��d�,�@)���ڐA2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<�������~�f^��`{�E'�/��x��i����v��
L�v�Q�28���FP���Ő�;F��`�}6D�/�p`B�6NK��2>�so쩾4>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>:�AC)8g>ڿ�ɓ�i>=�.^ol>�i����v>E'�/��x>f^��`{>�����~>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�               @      ,@     �V@     �f@     @f@     �j@     �s@      @     h�@     ��@     ��@     {�@    ���@    ���@    ��@    ���@    ���@    ���@    @��@     ��@     {�@    ��@     0�@     �@    �j�@    ���@    �'�@    ���@    @��@    �5�@    ���@    `&�@    `��@    �a�@     R�@    �R�@    @B�@    `��@    ���@     
�@     <�@    `?�@    @(�@    @�@     e�@    �t�@     U�@    `��@    ���@    ���@    ��@    @~�@     ��@    @��@     ��@    ���@    ���@    @��@    ���@    ��@     s�@     ;�@     ��@    �T�@    ���@    @s�@    �>�@    ���@    ���@     �@    �g�@    ���@     ��@     ��@     ��@     �@     	�@     �@     s�@     ��@     ��@     ��@     ��@     z�@     Χ@     z�@     ��@     h�@     ��@     4�@     4�@     �@     ��@     |�@     ��@     �@     ��@     `�@     �@     Ї@     ��@      �@     �@     @�@     P~@     p|@     �w@     0x@     �t@     �q@     �q@     @o@     @j@     �i@     �f@      h@      b@     `b@      ^@      Z@      Y@      V@     @W@     @U@     �N@     �J@     �I@     �F@      K@      B@      G@     �D@      A@     �A@      ;@      ?@      ?@      6@      8@      :@      :@      1@      &@      4@      2@      "@      1@       @      @      @      @      "@      $@      @       @      @      @       @       @      @      @       @      @      @       @              @      @      @      @       @       @       @      �?       @              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?               @      �?      �?              �?              �?      �?      �?      @              �?      �?      @       @      �?       @      @      @      �?              @      �?      @              @      @      @      @      @      @      @       @      "@      $@      (@      *@      @      (@      &@      1@      9@      3@      5@      ;@      :@      7@      C@      @@     �A@     �D@     �F@     �L@      H@      N@      H@     �P@     �Q@     �X@     �X@      Y@      [@     @]@     @b@     �b@     `d@     �f@     �i@     �l@     �m@     �p@     @r@     �q@      y@     Pw@      |@     0@     �@     ȁ@     �@     ��@     0�@     ؉@     �@      �@     p�@     ȓ@     X�@     d�@     `�@     ��@     �@     $�@     v�@     ��@     H�@     ��@     �@     ܬ@     ү@     ٱ@     ��@     X�@     ��@     k�@     �@     ��@     ��@    �*�@    ��@    �@�@    ���@    ���@    �Q�@    �-�@    @��@     P�@    @s�@     ��@    @;�@    �;�@    ��@    @��@    @�@     ��@    ���@    `��@    ���@     ��@     ��@     ��@     ��@    ``�@    �w�@     ��@     9�@    @6�@    `�@    @$�@    ��@    ��@     ��@    @��@     !�@    �C�@    `��@    `��@     #�@    �R�@    ���@    �b�@    ���@     b�@    ���@    ��@    �*�@    @Z�@    @��@     �@     #�@    ���@    ���@     ��@    ���@    ���@    @��@     ��@     ��@    �p�@    @E�@     X�@     T�@     ��@     `p@     �^@     �J@      5@      @        
�
predictions*�	   ��=��   ��?     ί@! �x�r�7@)5U�7D@2�
�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�������iD*L��>E��a�W�>�h���`�>�ߊ4F��>>h�'�?x?�x�?�.�?ji6�9�?�S�F !?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?�������:�
               @       @      @      2@      D@     �N@     �V@     �^@      _@     �[@     @`@     �W@     �Y@     �Y@     @U@     �S@      R@      P@     �P@     �O@      H@     @P@     �F@     �G@     �B@     �C@     �E@      8@      ,@      :@      .@      (@      1@      *@      (@      .@      "@      $@      .@      $@      "@      $@      @      @      @       @      @      @      @      �?       @      @      @      @      @       @              �?      �?              �?              �?       @      �?      �?      �?      �?      �?      �?      �?      @              �?      �?              �?              �?              �?              �?              �?       @              @               @               @      @      @      �?      @      @       @       @      @      @      @      @      @      $@      @       @      ,@      @      $@      &@      ,@      *@      7@      4@      8@      2@      7@      6@      B@     �@@     �A@      @@     �E@      A@     �E@      F@     �K@      L@     �Q@     �L@     �I@     �R@     �Q@     @Q@     �S@      Q@     �V@     @S@     �M@     �Q@     �J@     �H@     �P@      L@     �K@     �D@     �C@      6@      >@      ;@      2@      .@      @      .@       @       @       @      �?      �?        jN��%      ��R	������A
*�K

mean squared error�Q?=

	r-squared���<
�3
states*�3	   ��b�    �@ @    ��NA!�ՎBN��@)�R��6A2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�w`f���n�=�.^ol�cR�k�e������0c�w&���qa�d�V�_�4�j�6Z�Fixі�W�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH�/�p`B�p��Dp�@�����W_>�p
T~�;�7'_��+/��'v�V,�2!K�R���J�RT��+��y�+pm��`���nx6�X� ��J>2!K�R�>�'v�V,>7'_��+/>u 5�9>p
T~�;>�`�}6D>��Ő�;F>28���FP>�
L�v�Q>H��'ϱS>��x��U>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              8@     �a@     @i@     �h@     �u@     �@     ȅ@      �@     ��@     �@    �|�@    �|�@     ��@    �@�@    ���@    ���@    ��@    �s�@    �!�@    ���@    ��@     e�@    ���@     8�@    ���@     ��@     ��@    �V�@    ���@    @��@    ��@     ��@    �`�@    ���@     	�@    @�@    `��@    ���@    �7�@    ���@    ���@    ���@    ���@    `�@    @��@    ��@     �@    @��@    `��@    �1�@    ���@     �@    �J�@    ���@    @��@    `��@    @��@    ��@    @Q�@    ���@    �s�@     ��@    �M�@     ��@     H�@     ��@    ���@    ���@    ���@     �@    �:�@    ��@     A�@     ��@     ��@     ,�@     �@     `�@     L�@     9�@     p�@     t�@     8�@     ��@     ��@     P�@     �@     �@     ��@     ��@     ��@     ��@     �@     @�@     ̒@     ��@     T�@     `�@     (�@     `�@     ��@     8�@     ��@      �@     �}@     �|@     �y@     �x@     �u@     pt@     s@     @l@      n@     @j@     `j@     �g@      f@     �c@     �b@     �[@     �^@     �\@      Z@     @U@     �U@     @V@      T@     @P@      K@      J@      K@     �G@      D@     �A@     �@@      D@     �@@      @@      0@      9@      0@      1@      8@      2@      ,@      &@      $@      *@      $@      ,@      @      (@       @      &@       @      @      @      "@      @      @      @      @      @      @      �?      @       @      @      @              @      @              @      @      @      �?      @      �?      �?      �?      �?               @               @       @       @              �?              �?              �?       @      �?               @              �?               @              �?              �?              �?              �?              @              �?              �?              �?      �?      �?              �?      �?      @      �?      �?              �?      �?               @      @      @      @              @      �?              �?      �?               @      �?       @      @      @      @       @      �?      �?      @      $@      @       @      @      @      @       @       @       @      @      @      $@      &@      2@      .@      &@      (@      $@      7@      9@      .@      :@      .@      6@      5@      =@     �B@      @@      C@      F@     �M@     �I@      H@     �L@      N@     @U@     �U@     �U@     �W@     �^@     �W@     `b@     @b@     �c@      g@     �e@     �e@      i@     �p@      q@     �q@     �t@     �w@      y@     �x@     �|@     P~@     ��@     ��@     ��@     Ї@     ��@     �@     ��@     ��@     ԑ@     ȓ@     ��@     ��@     ��@     X�@     �@     ��@     V�@     X�@     $�@     ��@     ��@     \�@     ��@     ��@     ��@     b�@     ��@     ��@     !�@     ��@     ��@     ��@     $�@    �w�@     ��@     �@    �>�@     ��@    �!�@    �"�@    �*�@     ��@    @!�@     ��@    @��@    ���@    �y�@    `x�@    `r�@     T�@     �@    `�@     ��@    `;�@     ��@    �d�@    ���@    `��@    ���@     ��@    ���@    ��@    ��@    ��@    ���@    `��@    ���@    ���@    ���@    @��@    ���@    �C�@    @x�@    `(�@     ��@    `c�@     �@     Z�@    ���@    �z�@     F�@    @��@     d�@    @��@    ���@    �^�@    @�@     W�@    �v�@    @��@    �_�@     ��@     m�@    �'�@     ̹@     ��@     T�@     h�@     �h@     �Z@      =@       @        
�
predictions*�	   @P���   �wU�?     ί@! `*<��B@)���}��@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��f�ʜ�7
������6�]���1��a˲�O�ʗ�����Zr[v�����%�>�uE����>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��Zr[v�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?�������:�              �?      �?      �?      @      @      6@      :@     �M@     �K@      Q@      V@      Y@     @Y@     �\@      V@      U@     @U@     �U@      N@      R@      K@      L@      J@      H@      D@      F@      G@      =@      9@      ;@      8@      6@      7@      1@      .@      .@      4@      ,@      *@      0@      *@      *@      (@      @       @       @      @       @      @              @      �?      @      @      �?      �?       @      @       @      �?      @      �?      �?      �?      �?      @               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?      �?               @      �?      �?      �?      �?      �?       @               @              @      @      �?      @      @      @      @      @      @       @       @      @      @      &@      "@      @      @       @      ,@      6@      4@      0@      @      *@      :@      7@      >@      @@      >@      D@      C@     �E@     �I@     �J@     �L@     �O@     �Q@     �R@     @Q@     �V@     @R@     �R@     �P@     �U@      S@      S@      P@      O@     @S@     �Q@      M@      J@      O@      G@     �C@     �A@      A@     �B@      <@      4@      0@      ,@      &@      0@      &@      (@      @       @      @      @      �?      �?              �?        W��(      $��	ba����A*�O

mean squared error	�>=

	r-squared��<
�8
states*�8	    w��   ��@    ��NA!MY�&��@)���4��A2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q���Ő�;F��`�}6D�/�p`B�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,���-�z�!�%������f��p�Łt�=	���R����y�+pm��mm7&c��tO����f;H�\Q��i@4[���Qu�R"��Į#�������/��=��]���=��1���=nx6�X� >�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>�i
�k>%���>�'v�V,>7'_��+/>_"s�$1>6NK��2>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?     �J@     �i@     Ps@     �y@     Py@     �~@     (�@     ��@     �@     |�@    ���@    �V�@    ���@    ���@    @�@     k�@    �<�@    �W�@    ��@    ���@     �@    ���@    @��@     ��@    ���@    ���@    �A�@    ���@    ���@     8�@    `B�@    ���@    �y�@    �P�@     $�@    �V�@     <�@    @�@     �@    @��@    ���@    ���@    ���@     ��@    ���@    `��@    ���@     ��@    `��@    ��@     ��@     �@    �I�@    ���@    ���@    ���@    ���@    �&�@    �_�@     ��@    @��@    ���@    @��@    @��@    ���@     ��@     ��@    ���@    ���@     ��@     <�@    �b�@     ��@     #�@     �@     +�@      �@     =�@     Ĵ@     j�@     �@     �@     6�@     �@     R�@     ��@     ��@     >�@     ڠ@     �@     `�@     �@     t�@     X�@     ��@     $�@     �@     Ќ@     ��@     H�@     ��@     0�@     `�@     ؀@     �~@     �{@     �x@     �w@     �v@     pt@     s@     �q@     �p@      k@     �i@     @j@     �g@     �b@     `d@     �a@     �a@     �^@     @Y@     �Y@      \@     �S@     @T@     �Q@     �Q@     �K@     �I@     �I@      E@      E@      :@     �B@      A@      7@      8@      =@      7@      8@      3@      4@      9@      2@       @      "@      *@      @      "@      &@      @      (@       @      &@      @      @      @      @      @      "@      @      @      "@      @      @      @      @      @      @      @      @      @      @      @       @      �?       @      @       @       @              �?      �?      �?       @       @      �?       @      �?       @      �?      �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?              �?              �?      �?       @               @              �?      �?               @      �?              �?              �?               @      �?              �?      �?              �?              �?      @      �?      �?       @      �?              @              @       @       @       @       @      @       @      �?       @      @      @      "@      @      @      @      @      @      @      �?      @      @      @      @      $@      (@      &@      @      (@      (@      $@      "@      4@      0@      2@      0@      9@      .@      6@      7@      ;@      9@      ?@     �A@      B@     �G@     �J@      G@     �K@     �M@     �N@     �P@     �N@     �T@      W@     �T@     �`@     @[@     @\@     �a@     @c@     �d@     �e@     �i@     �k@     �j@     `l@     �m@     �p@     �s@     �t@     �v@     pz@      {@     0|@     Ȁ@     P�@     ��@     �@     Ȇ@     ��@     8�@     8�@     H�@     ��@     \�@     Ĕ@     ��@     $�@     0�@     D�@     ��@     ��@     �@     �@     ©@     Ϋ@     ڮ@     ��@     w�@     k�@     ��@     {�@     ޹@     ¼@     ��@    �G�@     ��@     �@     ��@     ^�@     G�@     ��@     ��@     �@    @��@    ��@    @��@     O�@    ��@    �3�@    @��@    �	�@    @��@    ���@     t�@    `Y�@    `L�@    `��@     ��@    �m�@     ��@    ��@    `"�@    �@�@    ���@    ���@    ���@    �!�@    `Z�@    ���@    ��@    `|�@    ���@    `S�@    `=�@    ���@    �J�@    `��@    �h�@    �;�@    `:�@     ��@    ���@     �@    ���@    ���@    �c�@    @y�@    �/�@    @5�@    ���@    @w�@    ���@     �@     �@    @.�@     �@    ���@     t�@     4�@     8�@     8�@     s@     �a@      G@      "@      @        
�
predictions*�	   ��!��    ��?     ί@!  BiG�'�)�0��@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.���vV�R9��T7���x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��G&�$�>�*��ڽ>;�"�q�>['�?��>jqs&\��>��~]�[�>})�l a�>pz�w�7�>O�ʗ��>>�?�s��>6�]��?����?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?�^��h��?W�i�b�?�������:�               @      .@      ?@      N@     �^@     �`@     @g@      j@     �e@     �d@     �`@      a@     @[@     @^@     �X@     �Z@     @X@     �V@      S@     �S@      S@     �M@     �I@     �B@     �C@     �D@      E@      :@      5@      4@      6@      .@      0@      4@      (@      ,@      ,@      &@      $@      @      @      @       @      @      "@       @       @       @      @      @      �?       @               @       @      @      @       @       @      �?       @      �?      @              �?       @      �?              �?              �?      �?              �?               @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @              @      �?              @      �?      �?       @      @       @      �?              �?      �?      @      �?      @      @      @      @      @      @      @      @      @      $@       @       @      (@      $@      &@      "@      &@      $@      ,@      1@      6@      1@      0@      5@      7@      .@      3@      <@      ?@      3@      =@      D@      @@      <@     �F@      B@      4@     �G@     �H@      D@      ?@     �F@     �E@      B@      C@      B@      9@      ?@      4@      6@      :@      6@      4@      4@      0@      ,@      &@      *@       @      "@      $@      @      @      @      @               @        �v �B+      ��	�"�����A*�V

mean squared errorR�:=

	r-squared��E=
�?
states*�?	    pE�   �8w@    ��NA!q�s>U*��)�Q}���A2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�6��>?�J���8"uH���Ő�;F��`�}6D�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	�Z�TA[�����"�y�+pm��mm7&c��`���nx6�X� ��tO����f;H�\Q������%���K��󽉊-��J��/�4��ݟ��uy�z�����i@4[���Qu�R"�H����ڽ���X>ؽ��
"
ֽ�|86	Խ;3���н��.4NνG�L������6���8�4L���<QGEԬ����_����e���]��̴�L�����/k��ڂ�x�_��y�'1˅Jjw��-���q�        �-���q=\��$�=�/k��ڂ=���6�=G�L��=��.4N�=;3����=(�+y�6�=�|86	�=���X>�=H�����=z�����=ݟ��uy�=��-��J�=�K���=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>Z�TA[�>�#���j>�J>��R���>Łt�=	>��f��p>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?     �U@     @d@     `o@     0x@     ��@     ��@     H�@     ��@     ң@     ]�@     b�@     ��@    ���@    ���@     3�@    �r�@    ���@    ���@    @J�@    @�@     ��@    @K�@    ���@    �(�@    ���@    `,�@    ���@    ��@     |�@     ��@    `8�@    @��@    �7�@    ��@    �7�@    ���@    ���@    @F�@    ���@    ` �@    `��@    `�@    ���@    `a�@    @9�@    `��@    `*�@     ��@    ���@    @�@    �n�@    `^�@    �T�@    ���@    ���@     ��@    ��@     ��@     ��@    @5�@    ���@    ��@    @��@    @b�@      �@    ���@    �R�@     ��@    ���@    ��@     ��@     ��@     �@     S�@     �@     ��@     U�@     ��@     ��@     �@     I�@     *�@     Z�@     �@     �@     H�@     ��@     �@     ԟ@     ��@     ̘@     ��@     8�@     ��@     �@     @�@      �@     P�@     ��@     0�@     ��@     ��@     �@     H�@      �@     P|@     0|@     `x@      v@     �t@     q@     �r@     �p@     �l@     �k@     �j@     `j@      d@     �b@     �c@     ``@     �`@     �\@     @Z@     �Z@     @Y@     @S@     �V@     �U@      V@     @P@      K@     �O@      D@     �K@     �N@      H@     �C@      A@      ?@      A@     �B@     �@@     �B@      ;@      >@      8@      ;@      4@      3@      5@      3@      1@      (@      5@      .@       @      ,@      &@      @       @      &@      $@      ,@      @       @      &@       @      @      @      @      @      @      @       @      @       @      @      @      @       @      @      "@      �?      �?       @       @       @      @       @      �?      @      @      @      �?      @      @      �?               @      �?      @               @       @              �?              �?       @               @       @      �?      @      �?               @      �?              �?              �?              �?               @      �?              �?              �?      �?               @              �?              �?              �?              �?               @              �?              �?              �?              �?       @              �?              �?              �?              �?               @              �?               @              �?              �?              �?      �?              �?       @              �?       @              @               @      �?              �?      @      �?       @      �?              �?       @       @      @      @      �?       @      �?              �?      @              @       @      @      @       @      �?      �?      @      @      @      @      @      @      @      @      @      @      @      $@       @      @      @      �?      @      @       @      @      @      @      &@      @      "@      *@      @       @      &@      (@      0@      *@      &@      &@      .@      "@      1@      &@      4@      5@      6@      6@      5@      9@      5@      :@      A@     �@@     �@@      B@      C@     �C@      F@      G@     �E@     �H@      J@     �N@     @S@     �S@      R@     �P@     �V@     �W@     �Z@     �X@      [@     �_@     �a@     �`@      e@      e@     `c@      g@     @i@     �o@     `n@     �o@     �p@     t@      t@     pw@     �y@      y@     �}@     @@     �@     �@     ȃ@     Ȇ@     Ї@     0�@     ��@     ��@     �@     �@     ��@      �@     l�@     �@     (�@     ��@     �@     ��@     .�@     8�@     �@     ��@     ~�@     ��@     E�@     >�@     ��@     $�@     ?�@     2�@     k�@     �@     N�@    �j�@     ~�@     ��@    ���@     j�@    ���@     4�@     =�@     ��@    ���@     ��@    @��@    @��@    @R�@    @��@    @��@    ��@     ��@     ��@    @��@     ��@    �m�@    ��@     �@    `��@    @T�@    ���@     ��@    ���@    ���@    ���@     ��@     ��@    @��@    �8�@    �l�@    �x�@    @��@    ���@     ��@     .�@    `��@     �@    ���@    `��@    ���@    ���@    ���@    ���@    ���@    ���@    @	�@     ��@    ���@    @L�@    ���@    @�@     <�@    ���@     4�@    ���@     ��@     d�@     ��@     ��@     �~@     �q@      `@     �E@      @      �?        
�
predictions*�	   �W��   �>X�?     ί@! B7�B@)l�k���(@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��O�ʗ�����Zr[v���ѩ�-߾E��a�Wܾpz�w�7�>I��P=�>f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?�������:�
              �?      �?      @      (@      8@      I@     �X@      _@     �_@      e@     �c@     �a@     �^@     �W@     @Z@      Q@      P@      Q@     @P@     �G@      M@      D@      =@      =@      >@      >@      <@      A@     �@@      5@      .@      2@      8@      ,@      *@      6@      "@      $@      *@      (@      &@      @      @      @      @      @      @      @      @       @      @      @      �?      �?      @      @      �?      @      �?               @      �?      �?      �?              �?              �?               @      �?               @              �?              �?              �?              �?      �?      �?       @              �?      �?      �?              �?      �?               @      �?      �?               @      �?       @      �?       @       @      @      �?      @      @      @      @      @      @       @      "@      @      @      @      "@      (@      $@      *@      (@      ,@      "@      1@      =@      =@      6@      *@      :@      <@      @@      <@      ;@      C@     �C@      I@      <@     �J@      F@     �K@     �I@     �M@     �P@      Q@     �K@      Q@     �P@     �M@      Q@     �M@     @P@      P@     @P@     �H@      F@     �F@      C@      :@      F@      >@      3@      5@      4@      *@      $@      "@       @      @      @      @      @      �?      �?      �?        ��Ւ.      F<�	!:����A*�]

mean squared errora;=

	r-squared��6=
�E
states*�E	   ��2�   `E�@    ��NA!��I|<��)Q���A2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�i@4[���Qu�R"����X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�EDPq���8�4L���<QGEԬ�|_�@V5��y�訥�V���Ұ����@�桽�>�i�E�����:������z5��e���]����x�����1�ͥ��G-ֺ�І�̴�L�����-���q�        �-���q=z����Ys=�8ŜU|=%�f*=�1�ͥ�=��x���=!���)_�=����z5�=���:�=������=_�H�}��=�>�i�E�=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=�8�4L��=�EDPq�=���6�=G�L��=5%���=�Bb�!�=�
6����=��؜��=�d7����=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              3@     @Z@     `j@     �p@      }@     ��@     ��@     P�@     H�@     ��@     �@     f�@     �@     �@    @��@    @��@     ��@    @��@    ���@     .�@     ��@    ���@    �,�@    ���@    @��@     J�@    �x�@    �=�@     ��@     �@     ��@    @.�@    ���@    �)�@    `��@     ��@     )�@    �3�@    �6�@    ���@    �-�@     [�@     ��@    �]�@    @n�@    �w�@      �@    ���@    ���@    �=�@     ��@     %�@    ���@    ���@    �3�@    �_�@    @��@    ��@    ��@    @8�@    ���@     ^�@     ��@    ��@    ���@    @	�@    ���@    ��@     H�@    ���@     Z�@    �S�@    �m�@    ���@     ��@     �@     Y�@     ǻ@     =�@     ��@     ��@     ��@     �@     �@     v�@     @�@     T�@     z�@     �@     ��@     �@     �@     <�@     �@     D�@     ĕ@     ��@     ��@     �@     T�@     ��@     �@     ��@      �@     ��@     �@     ��@     �@     �~@     �{@     �{@     �u@      u@     �t@     0r@     �p@     0p@     �o@     @j@     �g@      h@     `f@     �e@     `g@      a@     @_@     @^@     �a@     @W@     @V@      \@      \@      S@     �P@     �U@      U@      L@      H@      L@      J@      C@      <@     �L@      B@     �@@      D@     �A@      >@      :@      @@      @@      =@      >@     �C@      3@      9@      4@      4@      1@      5@      ,@      3@      6@      6@      3@      (@       @      0@      4@      .@       @      @      &@      $@       @      .@      @      "@      @      $@      @      @      @      @      @       @      $@       @      @       @       @       @      @      @       @      @       @      �?      @      @      @      @      @      @      �?      @      @       @      @      @      @       @               @              @      �?      �?       @       @      �?              �?      �?      �?               @      @              @              �?       @      �?              �?      �?      �?      �?       @      �?       @              @              �?      �?              @              �?              �?              �?      �?      �?              �?              �?              �?      �?      �?              �?              �?      �?              �?              @      @      �?              �?              �?              �?      �?              �?      �?               @              �?               @              �?               @      �?               @               @       @      �?      �?              �?      �?              @              �?       @      �?              �?      @      @       @              �?              @              �?               @       @       @              �?              �?      @      @      �?      �?      �?              �?       @              @      @               @      @       @      @      @       @               @      @      @       @      @      @      @      @       @      @      @      @       @      @      @      @      @      �?      @      @      &@      @      "@      "@       @      &@      ,@      (@      ,@       @      ,@      $@      &@      "@      ,@      (@      &@      ,@      1@      (@      7@      1@      :@      6@      8@      0@      >@      5@      ;@      =@      =@      A@      C@     �@@     �F@      =@     �D@     �C@     �G@      K@      L@      N@     �L@     �L@     @Q@     @P@     �R@     �Q@      Q@     �T@     �S@      ]@     @[@     �\@     @]@     @\@     �`@     `b@      d@      f@      f@     �c@     @l@      m@      o@     �o@     �r@     �q@     �t@     `x@      w@      |@     �{@     �~@     ��@     8�@     `�@     8�@     `�@      �@     0�@     8�@     ��@     �@     d�@     ̔@     ؖ@     ��@     �@     ��@     �@     �@     8�@     �@     �@     >�@     &�@     ��@     ԯ@     b�@     �@     f�@     ��@     }�@     _�@     ��@     I�@    ��@    �[�@     ��@     3�@    ���@     ��@    �M�@     7�@    �
�@     f�@    �(�@    ���@    �J�@    �J�@    ���@    �A�@    ���@     ��@    ���@    ��@    @u�@    �8�@    ���@    @��@    @8�@     ��@     <�@    ���@    `��@     �@    ��@    ���@    `��@    `��@    @��@    `�@     ��@    ���@     ,�@     +�@    �$�@    �A�@    ���@    �8�@    �9�@     ��@    �s�@    @ �@    ���@     ��@     ��@    ���@    �P�@    ���@    �8�@    �o�@    ���@    �d�@    �$�@    ���@     �@     D�@     ��@     '�@     u�@     ��@     t�@     �@     ��@     �}@     �r@      _@      J@      &@      �?        
�
predictions*�	   ��,��   ����?     ί@! `�#�(�)KV3���@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�1��a˲���[��jqs&\��>��~]�[�>})�l a�>pz�w�7�>I��P=�>��Zr[v�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?�������:�              @      0@      C@      R@     �U@     �a@     �a@     �b@     �c@     `b@      `@     �b@     �W@     �[@      W@     @S@     �P@     �L@     �J@     @Q@      O@      J@     �L@     �H@      ?@     �C@      H@      >@      B@      ;@      >@      3@      1@      *@      .@      3@      ,@      &@      (@       @       @      &@      @      @      @      @      @      @      @       @      @       @       @      @      �?      @      @      @      �?      �?              �?               @      �?              �?              �?       @              �?              �?              �?              �?              �?              �?      �?      �?       @       @      �?       @       @       @      �?               @      �?      @      @      �?      �?      �?      @      @      �?      @      @      @              @       @      $@      $@       @       @      "@      @      @      *@      ,@      1@      (@      *@      (@      9@      @      (@      6@      6@      =@      .@      ;@      A@     �A@      D@      A@      =@      E@     �B@      E@     �D@      F@      D@      I@      H@     �F@     �B@      J@      F@     �F@      C@     �E@     �A@      B@      @@      B@      8@      C@      7@      4@      0@      .@      3@      &@      ,@      "@      @      $@      @      @      @      @      @      @              @      �?              �?        ���a�.      [X��	m�s����A*�]

mean squared errorò;=

	r-squared��/=
�F
states*�F	    �C�   @ @    ��NA!W�p���)mH0�j�A2�#h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6�����5%����G�L����Į#�������/��<QGEԬ�|_�@V5��V���Ұ����@�桽_�H�}��������嚽��s�����:��!���)_�����_����̴�L�����/k��ڂ�z����Ys��-���q�        �-���q=z����Ys=\��$�=�/k��ڂ=��x���=e���]�=!���)_�=����z5�=�>�i�E�=��@��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=��؜��=�d7����=�!p/�^�=��.4N�=(�+y�6�=�|86	�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�#             �@@      d@     �p@     �x@     ��@     ,�@     Ж@     Ԝ@     ��@     ��@     ��@    �0�@    ���@    @A�@    ���@    @\�@    �d�@    ���@    �f�@    @��@     0�@    �u�@    ���@    ���@    @r�@    ��@    ��@     ��@     �@     %�@    �(�@    ���@    @��@    ��@    �;�@     �@    `��@    @��@    ���@    `y�@    ���@     R�@     ��@    �^�@    ���@    �A�@    �;�@    @��@     T�@     �@    ���@    �I�@     ��@    ���@    `i�@    @��@    ���@    ��@    @��@    @U�@     ��@    ���@    �y�@    ���@    �K�@    @��@    �`�@    @��@    ��@     �@    ���@    �1�@    �!�@    ���@    ���@    ���@     ��@     μ@     ��@     ��@     j�@     ³@     ձ@     U�@     V�@     ��@     �@     Ц@     �@     ģ@     x�@     ��@     ��@     P�@     �@     P�@     T�@     0�@     ��@     d�@      �@     �@     ��@     ��@     �@     P�@     ��@     ��@     ��@     P}@     P{@     @y@     `y@     �t@     �t@     @t@     t@     �l@     @p@     �j@     �j@      i@      k@     �d@     �f@     �c@      ]@     �a@     �`@     �]@      _@     �W@     �[@     @Y@     @T@     @S@      V@     �W@      R@     �K@     �P@      Q@     �L@      N@      G@     �L@     �H@      I@      G@      D@     �A@     �H@     �B@      =@     �@@      4@     �B@      ?@      ?@      8@      3@      =@      3@      1@      >@      3@      (@      6@      6@      2@      1@      (@      *@      1@      0@      "@      *@      "@      .@      *@      ,@      (@      $@      @      "@       @      (@      "@      @      @      @      @      "@      @      @      @      @       @      @      @      @      @      @      @       @      @      @      @      @      @      �?      @       @      @      @      @      @      @       @      @       @      �?              @      @       @       @      @       @      �?              �?      �?      �?      �?      �?      �?       @      @       @      @      @       @      �?       @       @      �?       @       @              �?       @              @      �?       @       @      �?       @      �?       @       @      �?              @               @              �?               @              �?              �?              �?              �?              �?      $@      &@      �?              �?              @              �?              �?               @      �?      �?              �?       @      �?              �?      �?      �?               @              �?      �?       @      @               @       @      @      �?              @      �?              @      �?       @      @      �?      �?      �?      @      �?      @      @      @      @      @      �?       @      @      �?       @      @       @      @       @      �?      @      @               @              @      @      @      @      @      @       @      @      @      @      @      $@       @      @      @      @      @      @       @      @      &@      @      &@      @      @      @      (@      @      *@      $@      3@      &@      4@      &@      *@      2@      3@      2@      (@      ,@      0@      9@      &@      =@      5@      *@      <@      ,@      8@      >@      5@      :@     �D@      @@     �@@      @@     �C@      B@      G@      D@     �G@     �C@      I@     �J@     �L@      L@     �M@      L@      M@      J@     �R@     �Q@     �P@     �R@     �T@     �S@      W@     @W@      ]@     �Y@      ]@      `@      a@     �]@     �_@     �c@     `b@      f@     �g@     �j@     @k@      k@     �m@     q@     �q@     �s@     @t@     0w@     �v@     �x@     �x@     �}@     �|@     h�@     P�@     x�@     �@     ��@     x�@      �@     X�@     L�@     ��@     ��@      �@     ��@     X�@     h�@     l�@     �@     ��@     �@     ��@     ��@     ��@     D�@     ث@     b�@     �@     e�@     3�@     �@     �@     ��@     p�@     �@    ���@    �X�@    ���@     �@    ���@     ��@     ��@     ��@    �x�@    @��@     _�@    @�@    ���@    �Q�@    ���@    @��@    � �@    @��@    @��@    ���@    `��@    �{�@    @T�@    `�@    ���@    `8�@    `��@    ��@    @j�@    ���@    `��@    @�@    @��@    `��@    �w�@    ��@     ��@     (�@    @��@    ���@    @��@    `��@    ���@    @��@    ���@     ��@    @&�@    @��@     ��@    ���@    �(�@    ���@    ��@    �!�@     p�@    ��@    �(�@    @��@     ��@     �@    �i�@    ���@    ��@     ��@     ��@     [�@     ��@     ��@     ,�@     P�@     p�@     �v@      d@     �T@      :@       @        
�
predictions*�	   ����   `��?     ί@!  ~���%�)!o�X[!@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(���ڋ��vV�R9�>�?�s���O�ʗ����ߊ4F��h���`f����>��(���>�h���`�>�ߊ4F��>I��P=�>��Zr[v�>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�
               @      �?      0@      ?@     �J@      V@     @^@      c@     �d@     �d@     �_@     �_@     �`@     �\@      U@     �O@     @Q@      Q@     �R@      P@     �I@      G@     �I@     �D@      G@      >@     �C@      F@     �D@      ;@      @@      ?@      8@      0@      *@      .@      2@      .@      "@      (@      &@      "@      "@       @      @      @      @      @      @      @      @      @       @              @      �?      "@      @      �?      �?       @      �?      �?      @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              @       @      @              @      �?       @       @      @              @       @       @      �?      @      �?      $@       @      @               @       @      "@      @      "@      $@      2@      .@      9@      *@      :@      .@      8@      7@      B@      >@     �A@      @@      C@      D@     �B@      C@     �G@      I@     �C@      I@     �E@      G@      S@      I@     �J@     �D@     �O@     �H@      L@      G@     �D@     �F@      B@      @@      8@      6@      2@      5@      &@      .@      2@      *@      @      @      &@       @      "@       @      @      @      @      �?      �?      @       @       @      �?      �?        � ��0      &�l	G�ф���A*�a

mean squared error�i9=

	r-squaredг_=
�K
states*�K	   �O�   `>@    ��NA!c�����) �{0��A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���ȽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]���1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=̴�L���=G-ֺ�І=�1�ͥ�=��x���=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%               @     �L@     �j@     �t@     ��@     Ȅ@     ��@      �@     ��@     j�@     ��@      �@    ���@    ���@     ��@    ���@    �N�@     ��@     +�@    �{�@    �Q�@    ��@    � �@    �%�@    ���@     ^�@    �	�@     ��@    ���@    @}�@    @��@    ���@    �,�@     ��@    ��@    ��@     i�@    ��@    `��@    ���@    ��@    `\�@     ��@     @�@    �;�@    �$�@    �,�@    ��@     ��@    ���@     %�@    `��@    @��@    �B�@     ��@     ��@     �@    �7�@    @k�@     ]�@    ���@    �Y�@    @��@    @��@    �1�@    �{�@    ��@     m�@    ��@     ��@     ��@     ��@    ���@     ��@     ��@     0�@     ;�@     1�@     �@     ͷ@     ĵ@     0�@     ��@     ��@     2�@     ��@     ��@     �@     ڥ@     ��@     ��@     �@     d�@     �@      �@     �@     ��@     ��@     ��@     �@     l�@     ��@      �@     H�@     ��@     ��@     ؃@     ��@     Ё@      �@     �~@     �z@     �y@     �u@     �t@     `u@     �s@     `q@      n@      o@     @l@     �m@     �h@      f@     �h@     �e@      b@     �a@     �c@     �]@     �Y@     �]@     �U@     �Z@     @\@     �S@      V@     @W@     @Q@      V@     �U@      P@      P@     �J@     �J@      G@      F@     �H@     �H@     �G@     �M@     �C@      @@     �B@     �F@      A@      D@      9@      <@     �@@      =@      <@      ;@      ;@      8@      5@      ;@      =@      2@      0@      ,@      7@      :@      *@      ,@      $@      .@      2@      $@      2@      (@      &@      (@      &@      ,@      (@      (@      ,@      $@      &@      &@       @      @      ,@      $@      "@      @      @      @      &@       @      $@      @      @      @      $@      @      @      $@       @      @      $@      @      @       @       @      @       @      @      �?      @      @      @       @       @      @      @      @       @      @      �?              @      @              �?      @      �?      @      @      �?      @      @      �?              �?      @      �?       @       @       @      �?      �?               @       @       @               @       @               @              �?       @               @      �?       @              �?      @              �?       @              �?       @               @      �?      �?              �?      @      �?              �?      �?      �?              �?              �?      �?              6@      2@      �?               @              �?       @              �?               @               @              �?       @      �?      @       @               @      �?              �?      �?              @       @       @              �?      �?      �?      �?      �?       @       @       @              �?              �?               @               @      �?              @      @      @      @      @       @      @      �?      �?      �?      �?       @      @      @       @      @      �?      @      @      @              �?      @      @      @      @      �?      @      @      @       @      @      �?       @      @      @      @      @      @      "@      @      @      @      @      @      @      @      @      @      $@       @      @      @      3@      (@      @      (@       @      @      (@      @      &@      "@      0@      3@      *@      ,@      ,@      "@      "@      1@      .@      0@      9@      6@      (@      5@      B@      3@      ;@      :@      3@      :@      >@      E@      ;@      @@      =@     �E@     �A@     �D@      B@      B@      A@      F@      B@      K@     �I@     �F@      F@      H@      I@     �H@     �R@     �Q@      N@     �Q@      P@     @Q@     �S@      R@     �W@     �W@      S@     �X@      \@     �\@     �\@      ^@      _@     �_@     `d@     `b@     @d@     `g@     �j@     `g@      k@     �j@     0q@     pq@     q@     ps@     �t@     0v@     Pv@     `x@     �|@     |@     �~@     p@     �@     0�@     �@     ��@     ��@     p�@      �@     8�@     ؎@     ��@     ��@     ��@     X�@     ��@     ��@     ,�@     @�@     ��@     ��@     �@      �@     R�@     ��@     �@     ��@     ��@     ��@     �@     ��@     =�@     �@     ��@     s�@     ��@    �#�@     ��@     ��@    ���@    ��@     �@     v�@    �!�@    ���@    ��@    ���@    �Q�@     ��@    ���@     O�@    ���@    �
�@     ��@    ���@     ��@     ��@    �5�@    �P�@    � �@    @��@     l�@    `��@    ��@    @l�@    `j�@     q�@    `��@    ���@    ���@    �Z�@    `��@    �a�@    `��@    ���@    ���@    ���@    ���@    ���@    @��@    @�@    ���@    @��@    �"�@    @��@     ��@    @2�@    �c�@    @Z�@    @��@    @��@    @t�@    @��@    ��@    ���@    ���@    �?�@     ��@     ��@    ���@     @�@     ��@     <�@     l�@     ��@     `�@     0}@     �k@      ]@     �E@      @      �?        
�
predictions*�	   @�w��    U@�?     ί@!  �b�R=@)�H7m��.@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7����5�i}1���d�r�1��a˲?6�]��?>h�'�?x?�x�?��d�r?�5�i}1?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�������:�
               @      @      4@     �G@     �O@      X@      b@      b@      e@      _@     @a@     �Y@     �S@     �P@     @T@     @Q@      L@     �G@     �J@     �B@     �F@      D@      ?@      @@      :@      >@      3@      :@      4@      &@      9@      1@      2@       @      4@      2@      (@      @       @       @      &@      @      @       @      "@       @      @       @      �?      �?      @       @              �?      @      �?      @      @      �?      �?      �?       @      @       @      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?              �?               @       @               @       @      @      @      �?      @       @      @      @      @       @      "@      @      @      &@      $@      @      "@      2@      0@      0@      2@      (@      2@      0@      9@      ;@      <@      ?@     �A@      A@     �B@      D@      O@      C@     �L@     @R@     �R@     �S@     �O@     �I@      S@      Q@      P@     @R@      W@     �O@     �L@      N@      I@      J@      F@     �J@     �F@      A@      A@      7@      9@      1@      0@      $@      1@      @      &@      (@      @      @      @      @      @      @      �?      @      @      �?      �?      �?        \)<�R1      #}�a	?�*����A*�b

mean squared error!7=

	r-squared@��=
�J
states*�J	   `���   `,j@    ��NA!筹�Y���)��p���A2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x����G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=|_�@V5�=<QGEԬ=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%             �S@     �p@     `w@     |@     �@     @�@     h�@     ��@     .�@     v�@     ��@    ���@     F�@     ��@    ��@     ��@    �T�@    ���@    �o�@    ���@    @��@     x�@    @d�@    �d�@    @��@    @ �@     �@    @Q�@    @��@     z�@    ���@    ���@     .�@    @6�@    @��@     ��@    ��@    @��@     ��@    `��@    @��@     W�@     ��@    ��@    �'�@    `�@    @b�@    `O�@    �J�@    @<�@    @��@    @��@     0�@    ���@    ���@    �Z�@     ��@    ���@     �@    ��@    �n�@     ��@    ���@    ���@     ��@    �^�@    ���@    �;�@    �r�@    �Q�@     �@     ��@     <�@    �I�@    ���@     )�@     b�@     �@     s�@     ܸ@     �@     д@     ��@     �@     (�@     ��@     ��@     n�@     ��@     ��@     4�@     ��@     L�@     d�@     �@     ��@     ��@     (�@     H�@      �@     ��@     <�@     ��@     X�@     (�@     �@     h�@     ؃@     ��@     (�@     0~@     p}@     p{@     �x@      x@     0u@     t@     @r@     �q@     @q@     �p@     �j@     �l@     �i@     `h@     �e@     �f@      f@      d@     �a@     ``@     �[@     �^@     �V@     �[@      [@      Z@     �X@     �X@     �U@     �M@     �R@     �W@     �P@     �N@      Q@     �P@     �D@     �N@     �J@      L@      G@     �H@     �H@     �F@      ;@      A@      B@      @@      >@      C@     �B@      <@      A@      =@      8@     �A@      9@      9@      0@      :@      1@      5@      7@      8@      ,@      3@      6@      3@      .@      8@      3@      ,@      *@      (@      .@      "@      .@      *@      "@      .@      $@      0@      ,@      (@      .@      "@      @      @      1@      @       @      @      @      @      @      @      *@      @       @      @       @      @      @      @      @      @       @      @      @      @      �?      "@      @      @      @      @      �?      @      @      @       @      @      @      @      @      �?      @      �?      �?       @      @       @      �?      �?       @       @       @      @      @      �?      �?      @      @              @      �?       @              �?      @      @       @               @      �?      �?      @      @              �?      �?      �?      �?              �?      �?      �?       @              �?      �?       @      �?              �?      �?               @              �?              �?      �?      �?      @               @      <@     �@@              �?              �?       @      �?      �?              �?               @              �?              �?      �?               @              �?      �?               @              @      �?       @      �?       @       @       @              �?              �?              �?       @      �?       @       @      �?      @       @       @              @      @       @      �?      @       @       @      @      �?      @      �?      �?       @      �?      @       @      @      @      @      @      @      @      @      @      @      �?      �?      @       @      @      @      "@      @      @      &@      @      @      @       @      @      @      @       @      @       @      @      "@      $@      "@      ,@      @       @      &@      &@      (@       @      3@      *@      (@      "@      (@      ,@      3@      5@      :@      *@      2@      4@      1@      7@      <@      <@      7@      9@      :@      :@      =@      :@      9@      <@     �D@      ;@      =@      ;@     �D@      C@      D@     �G@     �H@      N@      E@     �L@      G@     �M@     @P@     �P@      O@     �N@      P@     �R@     �O@      S@     �R@     �U@     �S@     @X@     @R@     @X@      \@     �^@     �[@     �_@     @`@     �a@     �b@     �a@      d@     �d@      f@     @d@     �i@     �j@     �k@     `l@     �p@     �o@     @o@      s@     �s@     �t@      v@     �x@      y@      {@     P|@     �@     @�@     ��@     8�@     ��@     ��@     ȇ@     X�@     8�@     x�@     8�@     P�@     ̒@     ��@     H�@     H�@     d�@     ؛@     ��@     P�@     �@     �@     ��@     ا@     `�@     .�@     ��@     :�@     ��@     3�@     '�@     a�@     p�@     �@     ��@    �7�@     ��@     ��@     �@    ���@     "�@      �@    �&�@    �N�@    ���@    @=�@     �@    @��@    @��@    �R�@    ���@    @��@    ��@    �:�@    @�@    ���@     ��@     Q�@    @�@    ���@    �!�@    `��@    ���@    ���@    `��@    �T�@    �y�@    @_�@    ��@    ��@     ��@    �/�@    �E�@     _�@    �k�@    �U�@    @d�@     $�@    ���@    �G�@    ���@    ���@    ���@    ��@    @��@    @��@    ��@    ��@     ��@    ���@     ��@     ��@    ���@    �]�@     ��@    @��@    @��@    @,�@    ���@    �f�@    ���@     ϲ@     ,�@     ș@     �@     �@     0�@     �x@     �i@      X@      M@      @        
�
predictions*�	   �&���    i��?     ί@!  0�-� @)n^��7@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���5�i}1���d�r�x?�x��>h�'����[���FF�G ���(��澢f���侕XQ�þ��~��¾��>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�h���`�>�FF�G ?��[�?6�]��?����?��d�r?�5�i}1?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?��Z%��?�1%�?\l�9�?�iZ�?+�;$�?�������:�               @       @      &@      @@     �I@     @T@     �Y@     �`@     �c@     �a@      c@     �b@     �Z@      ^@     �Q@     �W@      U@     �Q@      O@     �P@     �M@      I@     �J@      I@      D@     �F@     �B@      C@      E@      ;@      >@      =@      2@      :@      &@      0@      (@      ,@      ,@      @      @      "@      @      @       @       @      @      @      @      @      @      @      @      @       @      �?       @      �?               @       @      @      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?       @               @              �?      @      @       @      �?       @      @       @       @      �?       @      @      @       @       @      @       @       @      &@      @       @      @      $@      "@      4@      ,@      *@      =@      0@      4@      <@      =@      9@      @@      =@      <@      >@      E@      H@      A@     �D@      L@      C@      P@      H@      H@     �G@     �J@     �D@     �B@      J@     �L@     �I@     �D@      D@      H@     �C@     �@@      A@      ;@      =@      ,@      4@      1@      6@      4@      ,@      @      &@       @      @      @      @      @      @      @       @      @       @      @               @       @              �?        ?; �2      �zf	oՉ����A*�d

mean squared error:�5=

	r-squared�Y�=
�L
states*�L	    ��   @�@    ��NA!K^�&U���)�Qw1
 A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І��/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              @     �c@      r@     �y@     0|@     x�@     X�@     ؎@      �@     ܠ@     �@     ��@    ��@    ���@     k�@    �>�@    @��@    @}�@    ���@    @b�@     ��@    @x�@    ��@    ���@    �~�@    �m�@     %�@    ���@     #�@    ���@    ���@    ���@    �k�@    ���@     ��@    ��@    ���@    @�@     ��@    @��@    @i�@    �}�@    �6�@     ��@    `N�@    ���@    @��@    ���@    `��@    @�@    `��@    @��@    `��@     -�@    `��@     *�@    �.�@    ���@     ��@    ���@    ���@    ���@    �x�@    @��@    ���@    ���@    @��@     1�@    @6�@    @�@    ���@    ���@    ���@     _�@     ��@     ��@     E�@    ���@     ��@     }�@     ��@     ض@     e�@     ��@     �@     $�@     h�@     �@     L�@     �@     v�@     P�@     ¡@     ��@     ��@     4�@     t�@     ��@     �@     @�@     Ē@     H�@     ��@     ��@     ��@     ��@     ��@     ؅@     ��@     x�@     p�@     0@      ~@     }@      {@     @z@     @x@      v@     �u@     pr@     �s@     @q@     �n@     @n@     @k@     �j@     �g@     `f@     �e@      e@      d@     `b@      b@     �b@     �\@      Z@      _@     @X@     @[@     �V@      Y@      V@     �W@     @W@      Q@      V@      Q@      P@     �N@     @P@     �M@     �O@      L@      K@     �P@      J@     �F@     �G@     �I@     �D@     �C@      H@      A@     �C@      :@      B@      B@     �A@     �@@      <@      D@      C@      4@      3@      :@      4@      >@      6@      8@      5@      3@      1@      ,@      $@      3@      9@      ,@      "@      (@      .@      "@      "@      3@      $@      .@      &@      &@      @      2@      1@      *@      *@      $@      @      (@      &@      @      &@       @      �?       @      @      @      &@      @      @      @      @      @      @      $@      @      @      @      &@      @      @      @      @      @      @      @      @      @      @      "@      @       @      @      @      @      @       @       @              �?      @       @       @              @      @       @      @       @      @      @      �?      �?      @      @      @      @      @      �?      @      �?      �?      @       @      @      @      @       @      �?       @      @      �?       @      �?      �?      @       @      �?       @      @      �?              �?      �?      �?      @               @              @               @      �?      �?      �?      L@     �B@       @               @              �?              �?              �?              @      �?      �?               @              @      �?      �?      �?      @      �?       @              @       @       @      @               @      @      @      �?      �?       @       @       @      @       @               @               @               @      @              @      �?      �?      �?      @       @      @       @      �?      @       @      @      @      @       @      @      @              @       @       @      @       @      @      @       @      @      @      @      @      @      "@      @      @      @      "@      $@       @      @       @      @      @      @      $@      @      @      $@      ,@       @       @      (@       @      "@      0@      "@      $@      &@      $@      $@      *@      .@      $@      7@      ,@      5@      1@      2@      5@      *@      7@     �@@      ;@      7@      >@      5@      :@      8@      9@     �B@     �@@      @@      1@      B@      ?@      F@     �B@     �B@      C@     �B@     �D@      D@      I@     �G@     �J@      G@     �I@     �D@      M@     �M@     �H@     �Q@     @R@     �Q@     �P@      N@     �S@     �X@     �U@      V@      Y@     �X@     @U@     �Y@     �[@     �^@     �]@     �^@     �c@      a@     �a@     �d@      e@     @d@      g@     @g@     �i@     �k@      l@     �l@     @j@     pq@     �p@     �q@     �r@     �t@      s@     �v@     x@     �y@     @{@     �}@     @@     �@     �@     P�@     ��@     ��@     p�@     ��@     `�@     ��@     x�@     X�@     ��@     ��@     ��@     �@     0�@     0�@     ��@     ��@     �@     �@     Ԥ@     ^�@     �@     H�@     ��@     ��@     i�@     �@     u�@     ��@     ��@     c�@     ʼ@     ��@    �9�@    ��@    ���@     ��@    � �@     ��@     P�@     �@    �D�@    �"�@    @��@     ��@    �>�@    @$�@     s�@    ���@    ��@    `)�@     )�@     �@    ���@    ���@     D�@    ���@    �A�@     ��@    @��@    ��@    @��@    ���@     L�@    ���@     ��@    ���@     :�@     ��@     ��@    ���@    ��@    ��@    `E�@    @��@     ��@    � �@     ��@    @
�@     �@    @?�@    @��@    @>�@    �Y�@     ��@    �d�@    �0�@    ���@    ��@    ���@     �@    ���@    ���@    @��@     ��@    ���@    �e�@    �5�@     (�@     ��@     Ԙ@     ��@     ��@     ��@     px@     �k@     @`@     �D@      $@       @        
�
predictions*�	    ʖ��   �d3�?     ί@!  %{�F@)Ri!
")@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'�������6�]����ߊ4F��h���`�0�6�/n���u`P+d���f����>��(���>I��P=�>��Zr[v�>O�ʗ��>x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?�������:�              @      @      1@      ;@     �L@     �R@     @X@      _@     �[@     @Y@     @Y@      W@     �]@     @[@     �U@     @S@     �N@      M@     @P@      N@      C@     �B@      =@      B@      >@      <@      2@      7@      1@      8@      "@      .@      0@      $@      @      ,@      *@      ,@      (@      @      $@      @      "@      @      @      @      @      @      @      @              @      @      �?      �?      �?       @      �?      �?               @      @              �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?               @      �?       @      @      �?              �?       @               @      �?      �?      @      @       @      @      @      @      @      @      @      *@      @      (@      1@      @      *@      *@      &@      &@      2@      "@      5@      :@      9@      9@      @@      >@     �F@     �A@     �D@     �D@      H@      M@      J@     �M@     �J@      O@     �N@     �U@     �L@     �J@     �R@      R@     �Q@     @T@     �O@      P@     @R@      H@     �M@     �G@      F@     �E@     �B@      >@     �@@      ?@      9@      9@      (@      5@      4@      *@      (@      "@      "@      @      @      @      @       @      �?      �?              @      �?      �?              �?      �?        ,S�ݒ0      '	��	qr����A*�a

mean squared error��5=

	r-squared辕=
�K
states*�K	   @�[�    |#@    ��NA!K��r'��)ܿD@�A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              @     �f@     �j@      r@     y@     x�@     �@     \�@     ��@     z�@     Ȭ@     z�@    ���@    ��@    ���@    ���@    ���@    ���@    ���@    ���@    �$�@    @��@    ���@     ��@    �)�@     ��@    �F�@    �Z�@    �Q�@    @'�@     �@     ��@    ��@    @w�@    ���@    �M�@    ���@     ��@    �]�@    @8�@    �H�@    ��@    @�@    @��@    �"�@    ��@    ��@    ���@    �%�@    ���@    ���@    `p�@    @��@    �1�@    `T�@    ���@    �2�@     ��@    ���@     .�@    ��@     �@    �v�@     }�@    ��@     @�@    ���@    @	�@    @��@    ���@    ���@     ��@     ��@    ���@    ���@     �@    �C�@     ��@     �@     E�@     �@     ѷ@     ׵@     Ƴ@     i�@     ,�@     L�@     ��@     L�@     Ҩ@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     �@     �@     ��@     `�@     ܐ@     ��@     x�@     ��@     h�@     x�@     ��@     �@     @�@     8�@     ��@     �~@     0�@     }@     `{@     �y@     `u@      t@     �s@     �s@     `r@     �p@     0q@     `n@     �m@      i@     �j@     �j@     �f@     �f@     �c@     �b@     �b@     �`@     @b@      ^@     �[@     �_@      ^@     �Y@      Z@     @^@      U@     @Q@     �U@      W@      U@      S@      V@     @Q@     �S@     @U@     @P@      K@      N@      I@     �J@      F@     �L@      I@     �H@     �I@      D@     �D@     �B@      @@      B@     �@@      F@      >@     �@@     �C@     �@@      @@      @@      @@      ,@      6@      :@      1@      6@      6@      7@      4@      3@      5@      8@      5@      6@      9@      (@      (@      .@      *@      3@      2@      0@      ,@      .@      .@      "@      $@      .@      &@      &@      "@      @      @      (@      @      $@       @      0@      ,@      @       @      @      @      @      @      @      @       @      @      @      @      @      @      @      @      @      @      @      $@      @      @      @      @      @      @      @      "@       @      @      @      @      �?      @      @       @       @      �?      @      @       @      �?      @       @      �?      �?      @      �?       @      @       @      @      @      @      @       @      �?      �?              @      @              �?       @              �?      @      @      @              �?               @       @       @      @       @       @      �?      @      �?      �?               @       @      R@     �L@               @       @              �?      �?       @       @       @              �?       @      �?       @      @              @              @      �?      �?       @      @      �?       @       @      @              �?      @      @       @      @      �?       @              �?      @      @      @      �?      �?      @      @       @      @      @       @      @      @               @      @      @      @      �?      @       @      �?      @      @      @      �?       @       @      @      @      @      @       @      @      "@       @      @      @      @      @      $@      @      @      @      @       @      (@      "@      @      $@      "@      &@      ,@      &@       @       @      @      .@      &@      @      @      .@      2@      1@      1@      .@      *@      3@      5@      ,@      0@      6@      9@      9@      <@      4@      ;@      <@      6@      ?@     �@@     �A@      @@      D@      E@      :@     �E@      C@      E@     �@@      A@      C@      D@      C@      F@     �E@      H@      L@     �H@     �L@      K@      L@     �K@     @R@     �O@     �M@     �R@     �S@      S@      V@      S@     �V@     �[@     �Y@      ]@     �Z@      U@     @Y@     @^@     @`@     �Y@     �]@      d@     `c@     �b@      d@     �e@      d@     @g@      j@     `k@     �i@      o@      m@     �n@     @q@     `s@      s@     �s@     �t@     t@     �w@     �w@     Py@     �{@     0~@     ��@     ��@      �@     ��@     ��@     H�@     ��@     ��@     8�@     `�@     �@     D�@     �@     �@     ܔ@     ؕ@     ؘ@     $�@     ��@     О@     f�@     ̡@     l�@     |�@     �@     R�@     ,�@     Z�@     )�@     �@     ��@     ��@     S�@     w�@     L�@     �@    ���@     �@    ���@     g�@     ��@    ���@    ���@    ��@    @��@    @�@    ���@    @��@    �0�@    ���@    ���@     ��@    @��@    `�@    ���@    ��@     ��@     ��@    ���@     ��@    �;�@    ���@    �W�@    `��@    ���@    �>�@      �@    ��@    `�@    ���@    �<�@    ���@     ��@    ���@     ��@    ��@    ��@    ���@    �,�@    ���@    �M�@    @��@     ��@     n�@    @��@     ��@     $�@    ��@     �@     ?�@    ��@    @�@    ���@    ���@    @z�@    �U�@     P�@     X�@    �F�@     ��@     v�@    ��@     �@     ��@     Ж@     ؍@     ��@     �{@      u@      c@     @T@      6@      @        
�
predictions*�	    wf��    �:�?     ί@!  �x��G�)Vu�/%@2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?2g�G�A�?������?�iZ�?�������:�
              �?      &@      B@      J@      T@     �X@     �]@     @c@     @f@      d@      d@     �c@      e@     �e@     `a@     �`@      \@     �[@     @[@     �W@      T@     �R@     @P@      O@     �I@     �I@      D@     �B@      =@      =@      ;@      <@      <@      *@      9@      ,@      2@      "@      (@      &@      $@       @      @      @      @      @      @      @      @       @      @       @      @       @       @      �?              �?      �?      @       @      @      �?              �?              �?      �?              �?      �?              �?       @              �?      �?               @      @       @              �?      �?       @      @      �?      @       @      �?      @      @      @      @      @      @      @       @      @       @      &@      @      &@      $@      (@      .@      ,@      .@      ,@      6@      1@      3@      :@      4@      2@      0@      <@      @@      7@      7@      =@      >@      C@      0@      4@     �B@      >@      9@      @@      8@      8@      ?@      <@      =@      ?@      6@      *@      5@      "@      1@      @       @      .@      $@       @      @       @      @      "@      "@      @      @      @      �?       @              �?              �?              �?      �?        ��122      �H�	��A����A*�d

mean squared error��0=

	r-squaredx�=
�L
states*�L	    1��   ���@    ��NA!>� ����)�6(<�A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%              2@      h@     �c@     �i@     �r@     @}@     X�@     ��@     �@     l�@     ��@     �@    @��@    ��@    ���@    �n�@    @t�@    ���@    ��@    ���@     ��@    �,�@    ���@     ��@    ��@    @��@     �@    @�@     {�@    @;�@    �#�@    ���@    @��@    �g�@    @�@    @��@    ���@    ���@    �\�@    `��@    ��@    `n�@    �a�@    �g�@    �^�@    ���@    `$�@     x�@    ��@    `K�@    �Z�@    `P�@    `&�@     ��@    @(�@    ���@    `�@    �p�@    ���@    ���@     ��@     )�@    �r�@    @w�@    ���@     ��@     ��@    ��@    �_�@    ���@    �:�@    ��@    ���@    �~�@     ��@    ���@    �S�@    ���@     )�@     ��@     b�@     ׸@     ��@     �@     ߲@     ��@     ��@     R�@     ʫ@     �@     \�@     �@     N�@     ��@     Ġ@     x�@     |�@     К@     ��@     @�@     d�@     T�@     �@     ��@     ��@     0�@     ��@     ��@     ��@     `�@     (�@     X�@     H�@     �@     `@     �@     �{@     �y@     Pv@      y@     �t@     `v@     0v@      u@      q@     `q@     �p@     `k@     �l@     @j@     �k@     �g@      d@     �f@      g@     �d@     �e@     �a@     �b@     �c@     �]@     �`@      `@     �[@      \@     @Y@      [@     �X@     �W@     @T@      R@     @S@     @T@     �T@      R@      K@      P@     @P@     �P@      J@      N@     �O@     �M@      O@      J@     �L@      D@      H@     �H@     �C@      <@      F@     �@@      B@      A@      :@      ;@      @@      A@      :@     �B@      =@      <@     �@@      ;@      <@      3@      0@      8@      4@      2@      ;@      1@      4@      3@      0@      1@      3@      1@      "@      6@      "@      ,@      4@      3@      1@      *@      $@      &@      *@      (@      (@      *@      ,@      (@      .@      $@      $@      *@      (@      (@       @      .@       @       @      &@      @      @      @      @      "@      $@      @       @      @      @      @      @      @      @      @      @      $@      @      @      @      @      @      @       @      @      @      @       @      @      @       @      @      @      @              �?      @      @      @      @      @       @       @      @      �?      @      �?      @      @       @       @      @      �?      �?      �?      �?      @      �?       @      �?              @       @      @       @              @      �?      �?      �?      @      @              @      @      �?     @Y@     @T@       @      �?      �?      �?      @       @      �?               @               @       @      �?      �?      �?      �?       @       @       @      @      @      @      @      �?       @      �?      �?      @      @       @      �?       @       @      @      �?      �?      �?       @      �?      @      �?       @      �?       @      @      @      @      @      �?      @      @      @       @      @      @      @      @      �?      @      @      @      @      @      @      @      @      @      @      @      @      @      @      "@      @      "@       @      (@      @       @       @      @      $@      @      &@      *@      $@      &@      @      @       @      .@      @      0@      2@      (@      *@      @      4@      1@      (@      5@       @      4@      5@      .@      9@      8@      1@      .@      .@      ;@      ?@      5@      1@      6@     �@@      0@      :@      <@     �A@      A@      =@      B@      B@      A@     �A@      @@      D@      =@      G@      C@     �E@     �G@      I@      G@     �M@     @R@     �C@     �L@      R@     @Q@      N@      N@     �T@      S@     �Q@      T@     �S@     �T@     @V@     @W@      Z@     �R@     �[@     �W@     �[@     �Z@     �_@     @_@     �[@     �`@     �`@     �`@     �b@     �a@      c@     �f@     �f@     �g@     �j@     �i@      k@     @g@     `p@     @l@     Pp@      o@     q@     @s@     �r@     `v@      w@     �x@     �w@     �w@     @z@     P}@     �}@     �@     ��@     ؂@     �@     @�@     ��@     ��@     h�@     p�@     `�@     4�@     ��@     h�@     ��@     ��@     h�@     Й@     X�@     4�@     ��@     l�@     T�@     j�@     ��@     ��@     ��@     $�@     <�@     ��@     �@     f�@     ��@     ��@     �@     W�@     s�@    ���@    ���@    ���@    ���@    ���@     ��@     T�@    @�@    @��@     ��@    @h�@    @H�@    �P�@    �_�@    @b�@    �X�@     Z�@    �\�@     �@    ��@    `��@    ���@    �:�@    @�@    ���@     N�@    �>�@    �[�@    `Z�@    �y�@     ��@    ���@    �n�@    `��@     ��@     �@     5�@    �g�@    ���@    ���@     r�@    @��@     l�@     ��@    ���@    @R�@    ���@    ���@    �#�@    ���@    @��@     �@    ���@    @��@    ���@    @]�@    @��@    ��@     ��@     ��@    ���@    @|�@     ��@    ��@    �f�@    ���@     Ӳ@     �@     l�@     ��@     ��@     P{@     `r@      g@     �S@      =@        
�
predictions*�	    �.��   `��?     ί@!  ��d2A@)�Bzc.@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.���T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[��})�l a��ߊ4F���_�T�l�>�iD*L��>8K�ߝ�>�h���`�>�FF�G ?��[�?1��a˲?6�]��?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?2g�G�A�?������?S�Fi��?ܔ�.�u�?�������:�               @      @       @      &@      :@      ?@      D@      N@     �R@     �P@     �U@     �W@     �X@     @X@     �V@      T@     �W@     �M@      T@      S@     �N@     �P@      M@      M@     �M@      D@     �H@      E@     �A@      B@      ?@      >@      3@      4@      *@      3@      6@      *@      7@      *@      "@      @      @      "@      &@      "@      @      @      @      @      @      @      @      @      @      @       @      �?      @      @      @      �?      �?      @      �?       @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @      �?              @              @      @      @      @      @       @       @      @       @      @       @      $@      $@      "@      @      (@      &@      @      "@      *@      $@      (@      0@      2@      7@      0@      2@     �A@      7@      <@     �@@      D@      B@     �D@     �D@      D@     �O@     �O@      L@      J@      S@     �J@      Q@     �S@     @P@      N@     �R@      R@     �Q@     �K@     �J@     �K@     �F@      I@     �D@     �E@      8@     �@@      9@      ?@      5@      9@      :@      0@      $@       @      "@      @      @      "@      @      @      @      �?      @              �?      @      �?              �?              �?        �Z3      ���+	�������A*�e

mean squared error�6=

	r-squared��=
�L
states*�L	   �;��   �	�@    ��NA!U�[<��)I_�C��A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              $@     �d@     �f@     `n@     �u@     P�@     X�@     |�@     ��@     4�@     H�@    �.�@    ���@    ���@     ��@    �e�@    @n�@    ���@    @~�@    ���@    ���@    �7�@     q�@    �]�@    @��@    ���@     t�@     �@    ���@    �n�@    ���@    ���@    �B�@    ���@    @��@     n�@    �'�@    �b�@    ���@     ;�@    `��@    ���@    `��@     ��@     ��@    ���@     �@     V�@    @\�@    `!�@     ��@    `��@    �X�@     ��@     b�@     �@    �1�@     ��@    �#�@    �}�@     g�@    @a�@    �u�@    @��@     ��@    ��@    ���@     ��@     Q�@     ��@     ��@     �@     -�@    ��@     ��@    �$�@     z�@     ��@     �@     �@     @�@     �@     �@     N�@     ��@     �@     m�@     �@     ,�@     ��@     ֦@     ��@     d�@     Т@     ؠ@      �@     ��@      �@     �@      �@     Ȕ@     |�@     Ԓ@     T�@     ��@     ��@     ��@     Њ@     ��@     ��@     ��@     �@     ��@     �@     ��@     �@     �}@     �|@     �x@      x@     0w@     �v@     �t@     �s@     �q@     �q@     pq@      n@     �p@     �l@     `k@      k@      l@      h@     �f@     �c@     �f@     �c@      e@     �f@     �a@     �d@     �`@     �`@     @`@      a@     �[@     �[@     �[@      [@     �[@     �Y@     �W@     �X@     @U@     �U@     �S@      R@      Q@      O@      P@      S@     @R@      M@     @S@      T@      P@      L@      E@      G@      O@      G@      A@      C@     �L@      H@      F@     �D@     �D@     �C@      C@     �@@      B@      @@      D@     �A@     �@@      A@      >@      8@      6@      <@      ;@      4@      6@      8@      :@      2@      1@      .@      4@      (@      $@      5@      6@      .@      0@      .@      "@      &@      @      (@      @      &@      "@      @      5@      .@      *@       @      &@      "@      $@      $@       @      $@      @       @       @      @      "@      "@      @      @       @      "@      @       @      @      @       @      (@      @      @      @      �?      @      @      &@      @      @      @      "@       @      @      @      �?      @      @       @       @      @              @      @      @       @       @      @      @      @      @       @      @      @      @      @       @      @      @      @       @      @      @      @      �?      "@       @      @      @      �?       @      @      @      �?      @       @       @      @       @       @      @     �X@     @W@      @       @      �?      �?      �?      @      @      �?       @      @      �?      �?      �?      �?               @      @      �?      �?      @      @      @      �?               @      �?       @       @       @      �?       @      @      @      @      @      �?      @       @      @              �?      @      @      @      @       @      @      @      @      @      @       @      @      @      @      @      @      @      @      �?      @      @      @      @       @      @      @       @       @      @      "@      @      @      $@      @      "@      &@      @      @      @      @      &@      $@      .@      1@      "@       @      &@      ,@      (@      ,@      &@      .@      (@      "@      1@      .@      $@      .@      2@      5@      ,@      (@      7@      (@      8@      .@      .@      <@      :@      ;@      >@      2@      :@      ;@     �@@      G@      >@     �A@      =@     �@@     �G@      C@     �F@     �A@     �G@      D@      H@      C@      G@      C@     �I@     �H@     �K@     �P@     @R@     @P@     �O@     �Q@     �Q@     �Q@     �Q@     @R@     @S@     �Q@     �S@     @T@     �R@      T@      U@     @Y@     @[@     @X@     �`@      _@     �\@      a@     @_@     @_@     �b@     �d@      f@     �e@      i@     �f@      g@     �l@     @j@     �k@      j@     `n@     �k@      o@     �k@     @o@     �q@     pr@     0t@      s@     �v@     pv@     x@     �x@     p{@     �{@     �}@     P}@     �@     ��@     0�@     X�@     h�@     �@     ��@     �@     @�@     8�@     8�@     <�@     ܑ@     ��@     ̕@     D�@     ��@     x�@     ĝ@     ̟@     ֠@     ��@     ��@     ֥@     �@     ��@     l�@     ��@     \�@     v�@     C�@     �@     e�@     *�@     ��@     :�@     ,�@     ��@     S�@     ��@     ��@    ���@    ��@    ���@    �F�@    ���@    @9�@    ���@    ���@    �#�@    @��@    �{�@     ]�@    �H�@     Z�@    �U�@     a�@    ���@    ���@     p�@    ���@    ���@    ��@    �^�@     ��@    ���@    ���@    ���@    �J�@    ���@    ���@    ���@    `��@    ���@    `��@     0�@    ���@    �U�@    �&�@    `-�@    �%�@     ��@    @��@     ��@     ��@    @��@    ���@    ���@    @��@    �M�@     m�@    �b�@    @�@     ��@    ���@    @N�@     ,�@    �W�@    @�@     ��@     ��@    ���@    ���@     ��@     ��@     \�@     D�@     ��@     �~@     Pt@     �g@      T@      =@      @        
�
predictions*�	   ��gʿ   @U��?     ί@!  �l��7@)+�&���*@2��K?̿�@�"�ɿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
��������[���FF�G �O�ʗ�����Zr[v���ѩ�-߾E��a�Wܾ['�?��>K+�E���>��Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?x?�x�?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?3?��|�?�E̟���?�������:�              �?               @              �?      �?       @      @      (@      3@      6@     �D@     �F@      M@     �L@     @Q@     @U@     @Q@     �R@      R@      Y@     @T@      N@     �U@     @Q@     �R@      W@     �T@     �Q@      O@      J@      D@     �A@     �F@      B@     �A@     �@@      7@      8@      7@      0@      9@      6@      3@      0@      1@      @      "@      *@       @      &@      (@      &@      @      @      @      @      @       @      @      @      @       @      �?       @      @               @              �?              �?              �?              �?               @              �?       @               @               @              �?              �?              �?              �?              �?      �?               @              �?      �?      �?       @               @       @       @              �?               @      @      @      @      @      @      "@      @      @      "@      "@      @      (@      *@      $@      $@      &@      &@      $@      .@      8@      5@      2@      0@      .@      9@     �B@      @@      A@     �B@      A@      G@      H@      J@     @Q@      M@      O@     @U@     �Q@     �L@     @P@     �V@     �U@     �P@     �S@      M@      P@     �L@      G@     �J@     �G@      =@     �D@      ?@      ;@      7@      8@      4@      1@      2@      *@      &@      *@      $@      @      @      @      �?       @      �?      �?      @      @      @      @              �?       @      �?               @               @              �?        pa��R2      ��	�m����A*�d

mean squared error�J)=

	r-squared�>
�L
states*�L	    ���   `�^@    ��NA!<27;/��)����@A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              @     `g@     �b@     @i@     �p@     `w@     ��@     X�@     8�@     ��@     ֮@    ���@    �w�@     ��@     G�@     ��@     ��@    ���@    �-�@     �@    ���@    @�@    �>�@    @��@    ���@    @��@    ���@    @~�@    @��@    @�@    @��@    ���@    @��@    ���@     ��@    �<�@    `"�@    �A�@    ���@    ���@    �X�@    ���@    ���@    ��@    `��@     ��@    �0�@    �f�@    @z�@     s�@    @Q�@     �@    ���@     1�@    ���@    @�@     c�@    ���@    ��@    `k�@    �z�@    �Z�@    @$�@    �K�@    @?�@    �n�@    ���@    ��@    �m�@    ���@    ���@     .�@    ���@    ���@    ��@    ��@    ���@    ���@     ��@     ��@     +�@     ͷ@     ��@     8�@     �@     ��@     ��@     �@     ��@     *�@     ��@     P�@     �@     \�@     0�@     ��@     ,�@     0�@     ̗@     p�@     ��@     x�@     H�@     ��@     �@     ��@      �@     @�@     ��@     �@     ��@     8�@     �@     H�@     x�@     0@      |@      y@      {@      v@      w@     �u@     @u@     �t@     �r@     `s@     �r@     �m@     �o@     �o@     `l@     `k@     `i@     @k@     �k@     �e@      d@     �d@     �c@      b@     @b@     �e@     `a@     �b@     �`@     @^@      ^@      ^@     �\@      ]@      Y@     �X@     �U@     @V@     @T@     �V@      U@     �S@     @Q@      S@     �R@     �U@      S@     �T@     �Q@     �P@     �J@     �Q@     �M@     �K@     �K@      K@     �D@     �G@     �F@     �G@     �H@     �G@     �F@      A@      H@      >@      B@     �A@      :@     �A@      ?@      B@      E@      :@      =@      1@      :@      7@      9@      2@      2@      6@      8@      :@      4@      3@      2@      2@      4@      6@      *@      ,@      *@      6@      0@      4@      4@      1@      3@      5@      "@      &@      $@      $@      .@      @      @      "@      �?       @      0@      @       @      @      $@      *@      @      &@      @      (@      @      @      @      @      .@       @      @      @      @      @      @      *@      @      @      @       @      @      @      @      @      @       @      @      @      @      @      �?      @      @      @      @      @      @       @      @      @       @      @      @       @      @      @      @      @      @      @      @      �?      @      @       @      @      @       @      �?       @       @              @       @       @      �?       @       @      �?             �`@     �[@      @      �?              @              �?       @      �?      �?      @      @      @      �?      �?      @       @      @       @      @      @       @       @       @      @      �?       @      @      @      @      @      �?      @      @      @      �?      @      "@       @      @      �?      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      (@      @      @      *@      @      @      @      "@      $@      @      *@      @      @      @      @      (@      &@      $@      ,@      @      *@       @      (@      3@      .@      .@      0@      (@      3@      5@      1@      0@      2@      3@      :@      4@      9@      .@      2@      <@      .@      9@      7@      9@      <@      9@      :@      D@      2@      A@     �A@      C@      @@     �B@     �D@      B@      F@      ;@      C@     �A@     �C@     �F@     �K@     �N@      O@      I@     @Q@     �P@      J@     �M@      P@     �R@     @S@      N@      O@      P@     @P@      Q@     �R@     �P@     �R@     @[@     �U@     �Z@     @V@     �Y@     @W@     �W@     @Y@     �]@     �\@     �_@     �_@     �\@      [@     �e@     @`@      _@     @c@     �d@      c@     @e@     @f@     `f@     �i@      i@     �f@     `l@     �l@     �g@     @p@     @m@     0r@     @t@     �r@     @r@     �r@     @s@     Pu@     `t@     �u@      w@     @y@     0x@      {@     p}@     }@     (�@     ��@     �@     h�@     ��@     @�@     ��@     h�@     ��@      �@     0�@     |�@     �@     �@     L�@     ��@     ��@     @�@     h�@     ,�@     ��@     |�@     ơ@     >�@     ��@     ��@     ��@     �@     0�@     ^�@     �@     β@     ��@     ��@     �@     4�@     �@     u�@     ��@     '�@     �@     ��@     ��@    �H�@    ��@     �@    ���@    �f�@     n�@    @��@    �0�@    ��@     P�@    �u�@    �S�@    @��@     ��@    ���@    �z�@    �!�@    @��@     ��@    @4�@     ��@     ��@    �5�@     �@    �-�@    �
�@    �n�@     ��@    @3�@     .�@    `z�@    �U�@     t�@    ���@    `&�@    ���@     3�@     "�@    @��@    �0�@    ���@    �=�@    @��@    @t�@    @��@    ���@    @a�@    ��@    ���@    @m�@    �!�@     ��@    ���@     ��@    �U�@    @��@    ���@     G�@    ���@     ��@    @)�@     0�@     ��@     Ԛ@     X�@      �@     P�@     �x@     @k@      _@      @@      &@      �?        
�
predictions*�	   @����   `$@     ί@!  ��xg(@)?�dz�1@2�Ӗ8��s��!������%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�x?�x��>h�'�������6�]���>�?�s���O�ʗ������m!#���
�%W����d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?w`<f@�6v��@�������:�              �?               @      @      3@      B@      >@     �K@     �Q@     �Q@     �U@     �\@     @W@      W@     �[@     @W@     �U@     �X@     �Z@     �U@     �T@     �R@      Q@      P@      I@     �M@     �H@     �F@     �J@     �B@     �E@      8@      C@      7@      7@      2@      8@      3@      (@      $@      5@      *@      $@      "@       @      &@      @       @      @      @      �?      "@      @      @      @       @      @      @      @       @      �?      �?      �?      @      �?       @      �?              �?       @               @       @              �?              �?              �?              �?              �?              �?              �?       @              �?              @       @      @      �?       @      �?      �?      �?               @      �?      @      @      @      @       @      @      @      @      "@      @      @       @      "@      @       @      $@      *@      $@      1@      0@      (@      *@      0@      3@      3@      ;@      8@     �A@      7@     �@@      A@     �H@     �H@      J@      H@      G@     �K@      M@     �P@     �R@     @U@     �T@      O@      N@      I@      K@     @P@     �I@     �E@      K@      >@      A@     �@@      ?@      ;@      <@      .@      3@      1@      (@      &@      (@      @      @       @       @      @      @       @       @       @       @       @      �?      �?      �?              �?      �?              �?              �?        ���2      �zf	��\����A*�d

mean squared error�)=

	r-squaredB>
�L
states*�L	   ����   �'@    ��NA!�-p4�@)�Â;��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @     �`@     `h@      k@     Pr@     �y@     H�@     ��@     �@     4�@     �@     h�@    @l�@     d�@     k�@    �%�@    ���@     ��@     E�@     ��@    ���@    ���@    ���@    �Z�@     #�@    ��@     ��@    ���@     ��@    ���@    ���@    @B�@    @��@     ��@    �f�@     ��@     |�@    @/�@    �:�@    ���@    ���@    ���@    `��@    �?�@    �z�@    �!�@    ��@     ��@     ��@    ���@    ���@     ��@    `��@    �3�@    @��@      �@    ���@    @�@     �@     ��@     5�@    ���@     :�@    ���@    @�@    ���@    ��@    ��@     ��@     #�@     ]�@     ��@     Q�@     ��@     X�@     ��@     +�@     T�@     $�@      �@     .�@     j�@     ��@     ��@     ǰ@     x�@     N�@     "�@     �@     Υ@     ��@     2�@     ��@     �@     ��@     �@     t�@     @�@     �@     ��@     ��@     8�@     ��@     p�@     Ў@     ؋@      �@     ��@     P�@     p�@     ��@     X�@     `�@     ȁ@     ؀@     �@     �~@      |@     `{@     �y@     Pv@     �u@     �u@      u@     `t@     �r@     �q@     �p@      p@     �q@     �n@      m@     �i@      h@     @j@     �h@     `f@     �h@     �d@      f@     �e@     �b@     �b@     @b@      d@      `@     �_@      ]@     `a@     �]@      `@     �]@      \@     �X@     �V@      W@      Z@     �U@     �[@      X@     �R@      W@     @T@     �S@      U@      V@      R@     �S@     �P@     �M@     �N@      K@      I@      G@     �N@      G@     �D@      F@      G@     �D@     �B@      F@     �D@     �@@     �C@     �D@     �A@      B@      D@      ;@      A@      6@      =@     �B@      5@      @@      =@      >@      2@      ?@      7@      7@      &@      9@      ?@      0@      ,@      :@      6@      $@      0@      (@      0@      1@      ,@      ,@      1@      0@      1@      "@      3@      2@      ,@      @      $@      "@      &@      $@       @      &@       @      1@      @      0@      &@      ,@       @      "@      $@      "@      @       @      "@      @      "@      "@       @      @      "@      @      @      @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      �?      @      @      �?      @       @      @              @      "@      @      �?      �?       @      @      @      @      @      @      @      @      �?      @      @      @       @       @       @      @      @      @      @      @      @       @      @     @a@      \@      �?      @      @      @      �?      �?      @      �?      �?      @      @              @      �?      @      @      @      �?      @      @      @      �?       @      @      �?       @      @      �?              @      @      @      @      @      @      @      @      @      @      �?      @      @      @      @      @       @      @      @      @      "@      "@      @      $@      @      @      @      @      @      @      @      @      @      @      @      &@      @      .@      .@       @       @      &@      @      ,@      $@       @      @      5@      *@      7@      $@      3@      1@      .@      3@      0@      &@      3@      4@      0@      5@      1@      6@      *@      1@      3@      :@      9@      >@      7@      =@      ?@      :@      ;@      ?@     �@@      >@      =@      @@     �F@      H@      D@     �B@      E@      F@      A@     �J@     �E@     �E@     �L@      A@     @R@      H@     @P@     �M@     @R@     �N@      T@      L@     @Q@     �P@      S@      P@     �Q@     �R@     @S@      U@     �X@     �U@     @T@     @T@     �[@     �X@     �X@     �]@     �[@     @Y@     �X@     �X@     �Y@     �_@     @]@     �]@      ^@     �b@     �b@     �c@     �d@     �e@     �d@     �c@     `d@      i@     �h@      h@     �i@      i@      m@     @n@     `n@     �m@     `n@     `q@     0q@     0p@     @u@      t@     �u@     u@     �w@      w@     y@     �x@     @y@     �|@     |@     �}@     p�@     (�@     X�@     ��@     ��@     �@     ��@     H�@     ��@     �@     ��@     ،@     L�@     |�@     ��@     ��@     ԕ@     p�@     ��@     ��@     ��@     �@     Π@     �@     �@     �@     ��@     Ҩ@     >�@     ܭ@     ��@     �@     J�@     ��@     ��@     �@     �@     �@     �@     ��@    �5�@    ���@     ��@    ���@     q�@     ��@    �$�@    @��@    ��@    ���@    @g�@    �l�@    @C�@     O�@    �U�@     ��@    @��@    ���@    ��@    ��@    �]�@    ��@    �w�@    @L�@    ���@    �P�@     ��@    ���@    `��@    �J�@     ��@    `^�@    �`�@     0�@    ���@    `~�@    @I�@    �c�@    `��@    `x�@     g�@    �m�@     r�@    @��@    @��@     ��@    @1�@     z�@    ���@    ���@    ���@    �0�@    �R�@     ��@    @��@    ��@    @P�@    �_�@    ���@    ���@     ��@    �3�@    ���@    ���@     ,�@     �@     V�@     8�@     ��@     �@      x@      m@     �[@     �G@      ,@      �?        
�
predictions*�	    b쾿   ���@     ί@!  �(M�)^>P�6-@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�6�]���1��a˲���>M|Kվ��~]�[Ӿ��n����>�u`P+d�>pz�w�7�>I��P=�>6�]��?����?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?\l�9�?+Se*8�?uo�p�?�6v��@h�5�@�������:�              �?              @      @      *@      ;@      I@      Q@      Y@      `@     �d@     �c@     �d@     �j@     �h@     �e@     �f@     �e@     @b@      b@     @a@     �Z@     �^@     �W@     @W@     �Q@     @Q@     �K@      I@     �D@      A@      F@      ?@      4@      2@      4@      5@      0@      *@      (@      @      0@      @      &@      @      @      @      @      @      @       @      @      @      @      @      �?      @      @      �?      �?              �?              �?      �?      �?       @      �?              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?      �?      �?      �?      �?      @      @       @       @       @       @      @      @      �?      @      @       @       @      �?      @      @      @      @      @      $@      "@       @      @      *@      @      "@      ,@       @      "@      ,@      $@      5@      *@      5@      (@      5@      1@      6@      1@      5@      8@      5@      :@      >@      4@      4@      4@      7@      6@      ,@      6@      1@      ,@      1@      2@      (@      &@      "@      @      "@      @       @      @      @              @       @      @      �?              @       @      �?      �?       @      @              �?      �?              �?        DO��1      }�j�	�������A*�b

mean squared error�7=

	r-squared��=
�L
states*�L	   ����   ���@    ��NA!�G�L�n@)�G���$A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              $@     �`@      k@     pr@     pt@      ~@     ؆@     ȑ@     �@     8�@     ;�@     ��@    @��@    ���@     ��@    ���@    ���@    @�@    �8�@    @��@     ��@    ���@    �&�@    �_�@     ��@    ���@    ���@    �Z�@    �R�@    ���@    ���@    ���@    ��@    ���@    ���@    @��@    @Q�@    `��@    `��@    ���@    `��@    `��@    �h�@    �x�@    ���@     ��@    �	�@    @+�@    @T�@    ���@    @��@    ���@    ��@    @��@     ��@    @�@    ��@    �z�@    `��@    �N�@    @9�@    ���@     4�@    @U�@    ���@    @p�@    @��@    @x�@    ���@     s�@    �,�@    ���@     N�@     ��@    �v�@    ���@     ��@    �0�@    �6�@     @�@     �@     ��@     ��@     K�@     $�@     |�@     >�@     �@     *�@     @�@     ��@     8�@     \�@     �@     Z�@     J�@     4�@     �@     @�@     h�@     �@     �@     �@     ��@     ��@     �@     ��@     X�@     `�@     h�@     ؇@     �@     (�@     @�@     ��@     h�@      �@     �~@     `}@     �}@     `{@     Px@     �w@     �v@     �u@     �v@     �t@     �t@     @r@     `s@     �p@      q@     @m@     �k@     �i@     �i@     �f@     @h@     �f@      h@     @e@      d@     �c@     �d@     �e@      `@      d@     ``@     �_@      c@      a@     �a@     @[@     �[@      ^@      ]@      Z@      ]@      W@      W@     �X@      X@     @R@      X@     @S@     �T@      R@     �Q@     @R@      M@      J@     �O@      O@     �Q@      L@     �P@      L@     �G@     �I@      K@      K@     �H@     �I@      I@      E@     �B@      @@      ;@     �E@      8@      H@      >@      ;@      >@      <@      :@      =@      4@      5@      9@      :@      =@      ;@      <@      9@      8@      ,@      .@      4@      0@      3@      2@      0@      1@      0@      3@      "@      1@      4@      $@      &@      0@      3@       @      *@      $@      (@      .@      @      $@      &@      $@      &@      "@      @      *@      &@      @      $@      @      @      @      @      "@      @      @      @      @       @      @      $@      @      @      @      @      @      @      @      &@      @      @      "@      "@      @       @      @      @      @      @       @      @      @      @       @      �?      �?       @      @       @      @      @      @       @      @      @      @      �?              @       @      @       @      @      @      �?      @      �?      �?      �?       @      @      @      a@     �[@       @       @      @       @      �?       @      @       @       @       @      �?      @      @               @       @              @              @      @      @       @      @      @      @       @      @       @      @      @      @      �?      @      @      @      @      @      @              �?      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      "@      @      @      &@      &@      $@      (@      @      *@      @      5@      "@      &@      ,@      "@      *@      .@      0@      4@      2@      4@      0@      1@      :@      2@      0@      1@      ;@      (@      (@      ;@      6@      ;@      5@      @@      9@      2@     �@@      :@      ;@      9@      =@      8@      8@      C@      =@     �A@      E@      @@      A@      D@      F@     �F@      D@      F@     �K@      E@      G@     �N@      K@     �J@     �K@      J@     �N@      G@      P@     �O@      R@     �P@     �R@      M@     �T@     �S@      X@     �Y@     �W@     �Q@     �S@     �Z@     @X@      X@     �W@      Y@      \@     @Z@      \@     �[@     @a@      b@     @`@      `@     �^@     �d@     `a@     @a@      f@      d@     �a@      f@      g@     �i@     �h@     �i@     �j@     `m@     �m@     �l@     @p@     �q@     `n@     pq@     �r@     Pt@     pr@      u@     @v@     �u@     Pv@     `x@     }@     �{@     pz@     �|@     �@     �@     �@     ��@     ��@     ؃@     ��@     ȅ@     8�@     x�@     ��@     �@     ��@     ��@     h�@     ��@     ,�@     L�@     ��@     @�@     ��@     ��@     \�@     �@     ��@     ��@     $�@     Ħ@     ��@     <�@     Z�@     .�@      �@     ��@     (�@     ��@     Y�@     l�@     ٻ@     $�@     ��@    �7�@     ��@    ���@    ���@     ��@     ��@    ���@    ���@    ���@    ���@    @o�@     ��@    ���@     �@     ��@    ���@    �y�@    ``�@     V�@    ���@     \�@    `�@    @]�@     ��@    �(�@    ���@    ���@    ���@     �@    �2�@    `�@    `��@    @��@    @>�@    ��@    �c�@    @��@    `�@    ���@    `��@    ���@     ��@    ��@    �'�@    �S�@      �@    �S�@    �0�@    ���@     f�@    @�@    �z�@     (�@    @��@     .�@    @��@     �@    ���@    ���@     ��@    @��@    ��@    �.�@    ���@     V�@    �g�@     ��@     ή@     У@     ��@     ��@     ��@     �z@     �u@     �j@      Y@      =@      @        
�
predictions*�	    7�ܿ   ��v�?     ί@!  "v]_@)�V�U3@2�
��Z%�޿W�i�bۿ� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�1��a˲?6�]��?����?f�ʜ�7
?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�������:�
              �?               @       @      @      @      "@      (@       @      0@      .@      6@      *@      >@      3@      4@      B@      6@      .@      1@      =@      6@      1@      8@      &@      <@      6@      5@      2@      1@      &@      (@       @      1@      &@      ,@       @      (@      @      @      *@      $@      @      @      @      @      @      @       @              @      @       @       @      @              @              @      �?              �?               @              �?              @      �?               @              �?              �?      �?      �?               @               @       @      @      @      �?      �?      �?      @       @              �?       @       @      @      @      @      @      @      @       @      @      @      @      @      "@      "@      ,@      (@       @      &@      4@      1@      3@      5@      8@      @@      D@      B@      G@      E@     �M@     �G@      M@     �U@     @W@     �W@     @Z@     �`@     �]@     @b@     @e@      i@      i@      f@      i@     `h@     �c@     �`@      Y@     �U@     @T@      M@      Q@      =@      <@      6@      8@      2@      *@      &@      ,@      "@      "@       @      "@      @      @      @      �?      @              �?       @      �?              �?      �?      �?      �?      �?      �?        ?0�2      ���	Y����A*�e

mean squared errorɯ)=

	r-squaredx
>
�L
states*�L	    ���   �@    ��NA!��~M�(�@).R�*�rA2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              @      a@     `l@     �m@     0v@      �@     (�@     D�@     D�@     �@     �@     ��@    @]�@    �b�@     6�@     ��@    �Q�@    @9�@    @��@    �A�@    ���@     ��@    ���@    �G�@    ���@    ���@    @��@     ��@    ���@    �l�@    �$�@    �;�@    ���@    �h�@    `��@    @+�@    ���@    @e�@    `~�@    ���@    `2�@     ��@    ���@    ���@    �*�@    ���@    �q�@     ��@    @j�@    ���@     ��@    `"�@    `��@    ���@     	�@    �H�@    @��@     �@    �K�@    ��@    @��@    ���@    @��@    @C�@    ��@     ��@    ��@    �w�@    �1�@    ���@    �z�@     _�@     ��@     {�@     ��@     V�@     ��@     �@     e�@     ĺ@     B�@     �@     ��@     �@     ^�@     ��@     .�@     �@     ��@     ��@     z�@     Τ@     l�@     �@     ^�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     d�@     ̑@     `�@     �@     (�@     8�@     ��@     h�@     ��@     @�@      �@     P�@     �@     x�@     ��@     0~@     P}@      }@     �z@     �z@     0x@     �v@     �v@     v@     �t@     �s@     `q@      s@     pp@     �o@     q@     q@      n@     �k@     @j@     �i@     �g@     `h@     @d@     @d@     `d@     �d@     �e@      e@     �`@     �^@      b@     �`@     �_@     �`@     @\@      _@      `@     �^@      ]@     �Z@     @Y@     �U@     �X@     �S@     �R@     �W@      W@     �U@     �T@     @P@      R@     @W@     @P@      Q@      P@      K@     �L@      K@     �I@      Q@      P@      H@      O@      I@     �E@      D@      E@      L@      C@      >@      D@     �B@      B@      G@      C@     �@@      A@     �@@     �@@      1@     �C@     �A@      7@      :@      ;@      4@      >@      :@      4@      0@      ;@      1@      1@      2@      6@      7@      5@      1@      ,@      2@      5@      0@      4@      (@      *@      .@      &@      @      "@      @      ,@      .@      (@      $@      @      0@      (@      "@      @      (@      .@      @      $@      @      &@      &@      @      @      .@       @      @      @      "@      @       @      @       @      $@      @      @       @      @      @      @      @      @      @      "@       @      @       @      @      @      �?      @       @      �?      @       @      @      @      @      @      �?      @      �?      @      @       @      @      �?      @      @       @      �?       @      �?      �?      @       @      @      @      �?              h@     `d@      @       @      @       @       @      @      @      �?      @      @              @       @      @      @       @      @      @      @      @      @      @      @       @      @      �?      @      @      @      �?      @      @      @       @      @      �?       @      @      @      @      @      @      @      @      "@       @      �?      @       @       @      @       @      "@      *@       @      &@       @      .@      @      .@      "@       @      *@      &@      &@      *@      @      ,@      *@      (@      1@      0@      "@      &@      ,@      3@      ;@      1@      4@      *@      4@      5@      1@      2@      1@      4@      4@      2@      .@      =@      ;@      6@      7@      4@      ;@      4@      8@      =@      ;@     �B@      7@      D@      7@     �D@      D@      C@     �@@     �F@      H@      B@     �B@      =@     �E@     �G@     �H@      G@      M@     �M@     �B@     �Q@      N@     �L@     �K@     �L@     @R@      T@     �N@      M@     @S@      N@     �S@     �Q@      Y@      R@     �T@     �Y@      U@     @W@     �\@      Y@     �[@     @W@     @`@     �_@     @]@      a@      b@     �a@     �`@     �`@     �b@      d@     �a@     �d@     �e@     �d@     �h@     `d@      i@     �j@     �i@     �i@     �g@     �i@      m@      o@     �l@     0p@     �p@     �m@     �p@      s@     �s@     @s@     @t@      u@     �w@      x@     �x@     �z@     `{@     P@     p|@     ��@      �@     h�@     ؂@     ��@     �@     P�@     ؆@     ��@     H�@     ؊@     �@     ��@     ��@     ��@     ��@     �@     ��@     $�@      �@     �@     ��@     D�@     
�@     n�@     ��@     ܣ@     ԥ@     j�@     ��@     N�@     ��@     D�@     �@     �@     Ѵ@     ��@     �@     �@     ��@     "�@    �^�@     8�@    ���@    ���@    ���@    ���@     �@     K�@     [�@     p�@    @E�@    ���@    @��@    @��@     ��@    ���@     v�@    @�@    @*�@    `O�@    �<�@    �'�@    ��@    `��@    �>�@     o�@    �@�@    ��@     ��@    @)�@    ���@    @L�@    ���@    ���@    �+�@     ��@    @��@    ��@     ��@    `[�@    �>�@    `��@     ��@    ���@    �4�@     ��@     %�@    ���@     ��@    ���@     �@     �@     ��@    ���@    ���@    @��@    ���@    ���@    ���@    ���@    �	�@     t�@    @�@    �4�@     ��@     ��@     Ӻ@     ~�@      �@     �@     $�@     x�@     �@     0x@     Pq@     @h@      S@        
�
predictions*�	   ��/��   �m��?     ί@!  �C�`@�)^�;ē(@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x���FF�G �>�?�s���pz�w�7��})�l a���n����>�u`P+d�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?x?�x�?��d�r?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?yL�����?S�Fi��?�������:�              �?      @      @      1@      =@     �A@      L@     �P@     �R@      T@     �Q@     �V@     �W@     @X@     @X@     �Z@     �[@     �[@      \@     �\@     @Y@     @Z@     @]@      S@     �V@     @P@     @Q@      S@      L@     �Q@     �I@      I@      G@     �B@      @@      8@      A@      ;@      6@      1@      1@      ,@      (@      "@      &@      $@      *@      @      (@      @      @      "@      @      @      @              @      @      @      @      @      �?      �?      �?      �?      @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?       @       @      �?      �?      �?              �?              @      @      @      @      @      @      @      @      @      @      @      @       @       @      @      (@      "@      $@      &@      1@      0@      $@      .@      .@      0@      0@      7@      @@     �A@      .@      ?@      0@      A@      ;@      C@     �K@     �A@     �J@      K@     �I@      D@     �H@     �E@      F@      I@      C@      G@      ?@      @@     �B@      ;@      7@      8@      ,@      "@      0@      *@      *@      $@      "@      "@       @      @      @       @       @       @      @      @       @               @              @      �?      @      @               @              �?              �?              �?              �?        ?�p22      �H�	�y����A*�d

mean squared error�:=

	r-squared�~Q=
�L
states*�L	   @`��   ���@    ��NA!��Ĩ�@)B�"��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              $@     `a@     �j@     �p@     �z@     ��@     @�@     $�@     ę@     ʣ@     �@    �7�@     x�@     �@    ���@    �Z�@    �\�@    ��@     x�@     ��@     N�@     ��@    �J�@    ���@     R�@    �0�@     G�@     ��@     ��@    �|�@    ��@     ��@    @��@    `F�@    �1�@    @|�@     /�@    @?�@    �(�@     ��@     ��@    �=�@     m�@     p�@    `_�@    @l�@    ���@    �	�@    @��@    `~�@    ��@    ���@    `V�@    @��@    ��@    �~�@     ��@    @��@     :�@     V�@     
�@    �$�@     ��@     [�@    �{�@     ��@     p�@    @ �@    @��@    ��@    �$�@    � �@     ��@    ���@     ��@     $�@     ��@    �(�@     �@     Ƽ@     ݺ@     j�@     ��@     ��@     i�@     ϱ@     ��@     ��@     x�@     ��@     H�@     b�@     b�@     J�@     ��@     ��@     d�@     ��@     @�@     ��@     \�@      �@     ��@     ��@     ؐ@     ̐@     (�@     ��@     X�@     H�@     ��@     P�@     ��@      �@     ��@     ��@     ��@      ~@     �z@     @~@     {@     �y@      y@     �x@     �v@     �v@      w@     �r@     �r@      r@     �q@     pr@     �p@      q@     @m@      o@      j@     �i@     �g@     �j@      m@     �i@     @f@     �g@      c@     �c@     �c@     �`@     �b@     �a@     �a@     �b@     �\@     �^@     �^@     @]@      ]@     �Y@      [@     @_@     �V@     @U@     �Y@      \@     @X@     �W@      W@     �T@     �X@     �U@     @T@     �M@      S@     �S@      L@     �P@     �I@     �H@     �J@     �N@      P@     �E@      M@      L@     �E@     �K@      K@     �H@      H@     �H@     �F@      F@      C@      G@     �B@     �B@     �C@      F@      E@      8@      B@      C@      9@      B@      5@      <@      4@      8@      5@      ;@      4@      4@      5@      9@      2@      1@      *@      ,@      3@      *@      ,@      $@      2@      "@      0@      ,@      *@      4@      0@      $@      $@      (@      ,@      "@      $@      @       @       @      "@      @      (@       @       @      @      *@      "@       @      $@      @      @      "@      @      @       @       @      @      @      @      @      @      &@       @      @      "@      @       @       @      @      @      @      @      �?      @      @      @      @       @      @      @      @      @      @      @       @       @      @      @      �?       @       @              �?      @       @      �?      @      @      @      @      @       @      @     �a@     �b@      @      @       @      @       @      @      @      @      �?      �?      �?      @       @      �?      @       @       @      @       @      @       @      @      @      @      @      @      @       @      @      @      @      @      @      @       @      @      @      @      @      @       @      @      "@      @      @      @      (@       @      $@      @      @      $@      @      0@      @      "@      $@      $@      0@      $@      (@      @      @      (@      @       @      ,@      0@      3@      .@      ,@      &@      2@      4@      7@      2@      9@      *@      ;@      4@      2@      9@      7@      7@      <@      1@      2@      1@      <@      8@      >@      9@      2@      ?@      5@      E@      ;@      =@     �A@      <@      =@      B@     �@@     �G@     �@@     �H@      B@     �B@     �H@     �G@      L@      G@     �J@      H@     �K@     �O@      L@      P@     �H@      M@      G@      R@      N@     �J@      R@      M@     @Q@     �U@      T@     @W@     @V@      R@     @T@     �W@     �U@     �Y@     @W@     �]@     @_@     �^@     @]@     �`@     @a@     @]@      a@     �^@     �b@     �a@     �]@      c@     �c@     �e@     �c@     @e@     �f@     @e@     �g@      k@     @g@     �i@     `i@     �h@     `j@      n@     `n@     �q@     �p@      r@     �q@     �p@     �r@      u@     �v@     Px@     �t@     pv@      v@     �x@     P{@     �}@     P~@     �@     ȁ@     ȁ@     0�@     ��@     ��@     h�@     P�@     ��@     ��@     �@     ��@     H�@     ��@     T�@     �@     Ȓ@     ��@     ��@     $�@     �@     L�@     (�@     ,�@     ��@     �@     V�@     ֤@     ��@     "�@     ��@     ��@     ��@     ��@     q�@     ,�@     ��@     ��@     ��@     �@     �@     ��@    ���@     ��@     ��@    �L�@     �@     @�@     
�@    ���@    ���@     ��@    �8�@    ��@     ��@    �+�@     �@    @N�@    ���@    ���@    �P�@     Y�@     x�@     ��@    `g�@     ��@     s�@    ��@    ��@    �+�@    `��@     ��@    @�@    ���@    @4�@     ��@    @��@     ��@    `��@    �z�@    �~�@    ���@    �2�@    �t�@    �H�@    �k�@    �T�@    �_�@    ���@    ���@    �l�@    @}�@    �l�@    �2�@    ���@     ��@    ���@     c�@    ���@    @��@    ���@     ��@    ��@    �E�@    ���@     ��@     ��@    ���@    �#�@     �@     r�@     �@     L�@     ��@     ��@     ��@      z@     �r@     �m@     @[@      &@        
�
predictions*�	    lR��    ��?     ί@!  @�B#�)��'�'%@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�x?�x��>h�'��O�ʗ�����Zr[v����(��澢f����jqs&\��>��~]�[�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>6�]��?����?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?2g�G�A�?������?cI���?�P�1���?�������:�              �?      @      @      .@      4@      E@     �N@      K@     @S@     �W@      T@     �Z@      T@     @U@      Y@     �S@     @R@     �Q@     �S@     �T@     �S@     �N@     �S@      O@      P@     �Q@     �P@     �K@      N@     �F@     �H@     �G@      B@      @@     �C@      0@      C@      9@      4@      6@      1@      3@      ,@      3@      .@       @      &@      "@      @       @      "@      @      *@      @      @      @      @      @      @      @              @      @       @              �?              @               @      �?               @               @              �?              �?              �?      �?              �?              �?               @              �?       @       @       @              �?               @              �?       @       @      @      @      @      @       @      @      @       @      @       @       @      @      &@      @       @      .@       @      2@      ,@      2@      .@      4@      2@      *@      7@      5@      A@      D@      @@      B@      D@     �A@     �I@      L@      B@      I@      H@      L@     �M@     �O@      N@      O@      N@     �P@     �N@      L@      I@      F@     �G@     �@@      6@      =@      ;@      3@      2@      7@      (@      (@      *@      &@      $@      @      @      @      $@      @       @      @      �?       @       @              �?       @      �?      �?              �?              �?        �X��r2      �$A	��ֈ���A*�d

mean squared error�l,=

	r-squared�v�=
�L
states*�L	   ��r�    *�@    ��NA!H�l�/�@)̦��1A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @      c@      c@     `l@     �u@     �@     ��@     �@     P�@     V�@     ^�@     |�@    ���@     �@    �m�@     E�@    @*�@     r�@    ���@     ��@    ���@    ���@     Q�@    ���@    �N�@     ��@     7�@     M�@    �"�@    @a�@    �1�@    ���@    ���@    �,�@    ���@    ���@    @~�@    @��@     ��@    ���@     ��@    @��@    @��@    ���@     ��@    �}�@     x�@     ��@    ���@     ��@    ���@    @Y�@     ��@    �.�@    ��@     e�@    ��@    `��@     �@     ��@    @��@    ���@    ��@     Y�@    �	�@    @E�@    ���@    ���@    �*�@     ��@    �q�@     ]�@     ��@     ��@    ��@    ���@    ��@    ���@    ���@     ��@     �@     ؽ@     ��@      �@     ��@     \�@     ��@     ڱ@     ��@     ̭@     ܫ@     ~�@     Χ@      �@     L�@     ��@     ��@     >�@     ��@     <�@     T�@     �@     d�@     d�@     0�@     H�@      �@     ,�@     ��@     ��@     8�@     x�@     H�@     h�@     ��@     ��@     ȃ@     ��@     ��@     0@     �@     �}@     �y@     py@      y@     Px@      w@     `t@     �s@     0t@     �r@     �r@     0r@     �p@     �p@      k@      k@      m@      n@      k@     �k@      j@      i@     �i@     �g@     @d@     �f@     �d@     @b@      a@     �^@     �\@     @_@     @`@     �`@     @`@      a@     �]@     �]@      ^@      W@     �Z@     �Z@     �Z@     �Y@      W@     �W@      S@     �S@     �Q@     @Q@     @S@     �P@     �N@      O@     @Q@     @Q@     �O@     �Q@      Q@      N@     �P@     �N@      O@     �L@     �L@     �E@     �H@     �J@     �E@     �B@      F@      G@      H@     �A@     �C@     �@@      F@      >@     �A@      >@      ?@      =@      <@      ;@      A@      ?@      ;@      <@      ;@      8@     �@@      9@      ?@      6@      1@      5@      6@      7@      3@      5@      5@      2@      .@      4@      ,@      &@      (@      0@      7@      2@      0@      "@       @      $@      (@      @      $@      1@      "@      $@      *@      (@      @      $@      &@      &@       @      "@       @       @      "@      (@      @      @      @      @      "@      $@      "@      @      @      �?       @      @      @      @       @      @      @      @      @      @      "@      @      @       @       @      �?      �?      @       @      @      @      @      @       @      @      @      @      @      �?      @       @      @      @      @      @      @      @      @      @     `i@     �e@              @      @       @      �?       @      @      @      @      �?               @       @      @      @       @      @      @      @      @      "@      @      @      @      @       @      @       @      "@      @      @      "@      @      @      @      *@      @      @      @      "@      "@      $@      @      @      @      @      @      @      $@      "@       @      (@      @       @      &@      (@      *@      "@      ,@      @      @      ,@      *@      .@      *@      (@      1@      ,@      ,@       @      "@      0@      *@      1@      6@      5@      *@      0@      @      9@      7@      1@      3@      8@      ;@      4@      3@      8@      =@      5@      @@      6@      C@      1@     �F@      A@      ?@      E@     �A@     �@@      ?@      C@     �F@      H@      B@     �B@      J@      G@      G@     �K@      G@     �D@     �E@     �O@      I@      N@     �J@      I@     �M@     �N@     @Q@     �N@     �O@      O@     �R@     �U@     @R@     �W@      Q@     �T@     �S@     @V@     �X@     �X@     �\@     �\@     @W@     @W@     �X@     �`@      ^@     @]@      \@     �[@      ^@     �^@     �`@     �b@      f@     `c@     �c@     �f@      f@     �d@      g@     @j@     @f@     `h@     �k@     �j@     �j@     �j@     @l@     pp@     �p@     Pp@      q@     �o@     s@     �s@     �s@     u@      u@     �w@     �w@     @y@     �z@      y@     �}@     �~@     �@     ��@     P�@     ��@     ��@     ��@     p�@     `�@     ؉@     X�@     ��@     �@     ��@     |�@     ��@     ��@     ,�@      �@     �@     ��@     ��@     �@     ��@     ȟ@     ��@     >�@     X�@     H�@     �@     T�@     �@     ­@     `�@     [�@     ��@     %�@     �@     η@     a�@     w�@     �@    ���@    �(�@     ��@    ��@     ��@    ���@     ��@    ���@    ���@    ��@    �k�@    @��@    ���@    @��@     ��@    �T�@    @��@    �)�@    �+�@    ���@     ��@    @^�@    ���@    `a�@    ��@    �e�@     ��@    `��@    @�@    �e�@    �Q�@    `z�@    @x�@     D�@    @��@    ���@    ���@    ��@     ��@     B�@    ��@    �I�@     6�@    ��@    �E�@    ���@    @��@    ���@    ���@     ��@    @$�@    ���@    @H�@    �7�@    ���@     ��@     ��@    ���@    @��@    @��@     ��@    ���@    @�@    �w�@    ���@     +�@     T�@    ���@     ��@     ��@     :�@     Ԯ@     �@     ��@     ��@     Љ@     �}@     ps@     �h@     �`@       @        
�
predictions*�	    �4ؿ   ���@     ί@!  +U��Q@)N���4@2��^��h�ؿ��7�ֿ� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��a�Ϭ(���(����uE���⾮��%�>�?�s��>�FF�G ?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?cI���?�P�1���?3?��|�?�E̟���?�6v��@h�5�@�������:�              �?               @       @      $@      0@      1@      5@      @@      D@     �G@      L@      A@      D@     �G@     �L@      G@      I@      F@      E@      H@     �C@      D@     �C@      C@     �C@      D@     �@@      B@      3@      9@      8@      <@      8@      2@      3@      .@      $@      0@      @      $@      &@      @      @      @      @      @      @      @       @      @      @      @      @      @      @              @              @       @       @       @              �?      �?       @      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?              �?      �?              �?      �?      �?      @       @       @       @      @      @       @      �?      @      @              @      �?      �?      $@      "@      @      *@      @      $@       @       @      @      $@      0@      2@      4@      0@      (@      6@      <@      ;@      8@      C@      C@      D@     �B@     �H@      J@      K@     �Q@     �W@      W@     @U@      [@      [@     �Y@      `@     �^@      `@     `c@     �]@     �[@     �\@      Y@     �R@     �S@     �M@     �M@      C@      C@     �@@      8@      3@      5@      ,@      *@      ,@      &@      @       @      @      @      @      @              @               @               @      @               @              �?              �?              �?        ��/�2      ����	��2����A*�e

mean squared error+�$=

	r-squared �$>
�L
states*�L	   �x��   `'�@    ��NA!�͖��@)�����A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              *@      a@     �b@      n@     `x@     ��@      �@     4�@     �@     �@     ү@    �R�@    @v�@    ���@     �@    �~�@     ��@    ���@    �Y�@    ���@     ��@     ��@    ��@    �{�@    ���@    ���@     ��@    @��@    �{�@    ���@    ���@     �@    @��@     l�@    ��@    @��@     ��@     ��@    @��@    �$�@    ���@     V�@     ��@    ���@    ���@     ��@    @��@     r�@     ��@    ���@     �@     ��@    ��@    ���@    �9�@    ���@    @q�@    ���@    �b�@     ��@    ���@    �X�@     u�@    @(�@    @��@    �9�@     ��@    �,�@    ��@    ���@    �.�@    ���@     ��@     ��@    ���@     V�@    �b�@     p�@    �Q�@     ��@     a�@     ��@     0�@     ��@     ��@     �@     F�@     J�@     &�@     P�@     D�@     8�@     8�@     ƣ@     �@     N�@     ğ@     ܞ@     ��@     |�@     �@     ��@     4�@     ē@     ,�@      �@     ��@     ��@     x�@     @�@     H�@     ��@     `�@     ��@      �@     0�@     h�@     Ё@     ؁@      �@     `|@     @}@     }@     �{@      y@     0y@     �v@      v@      u@     �t@     �q@     s@     @q@     @q@      q@     �p@     �n@     �p@     @n@      l@     �i@     �m@      j@     @g@      g@     �f@      d@     @c@     `d@     `a@     �c@     �c@     @a@     �`@      \@     �_@     �`@      _@     �]@     @[@     @X@      Z@     @Y@     @]@     �[@     �X@     �U@      X@     @U@     �T@     @U@     �T@     �Y@     �S@     @T@     @S@      R@      R@     @R@     @P@     @Q@     �O@     �M@     �L@     �Q@      H@     �G@     �G@     �D@     @P@     �G@      F@     �F@      N@     �A@     �G@      B@     �G@      C@     �H@      =@      7@      E@     �D@     �A@      A@     �B@      ?@     �A@      =@      >@      <@      3@      :@      8@      =@      <@      8@      4@      ;@      :@      :@      2@      3@      2@      1@      <@      2@      "@      (@      3@      (@      7@      .@      *@      ,@      ,@      @      *@      1@      &@      2@       @      (@      @      1@      $@      &@      ,@      &@      "@      (@      $@      .@      $@      "@      $@      @       @      @      @      "@      @       @      @      @       @      @      @       @      "@      (@       @      @      @      @      @      @      @      �?       @      @              @      @      @      @      @       @      @      @      @       @      @      @      @      �?      @       @      @       @      @     �i@      f@      @      @      @      @      @      @      @      @      �?      @       @      @      �?      �?      @      @      @      @      @      @      @      @      @      @       @      @      @      @      @      @      @      @       @      "@      $@       @      "@       @      $@      @       @       @      @      *@      ,@      @       @      ,@      "@      @       @      (@      @       @      $@       @      @      ,@      9@      $@      .@      5@      &@      &@      0@      *@      .@      &@      6@      0@      &@      ,@      "@      &@      ;@      9@      3@      ,@      6@      *@      4@      0@      8@      9@      :@      ?@      4@      >@      ?@      8@      @@      <@      =@      ?@     �@@      F@     �C@     �E@      B@     �C@     �A@      I@      D@     �E@      D@     �E@      G@      N@     �N@     �H@      H@     �G@      G@      N@     �H@     �M@     �Q@      M@     �J@     �R@     @T@     �T@     �O@     �Q@      R@      U@     �S@     �Q@     @S@     �V@     �W@      W@     �Y@     �Z@     @V@      [@      Y@     @[@     �]@     �\@     �b@      \@     `a@     �c@     �a@     �\@     @^@     �`@     @f@      e@      e@      h@     �f@     �h@      j@      h@     �k@      i@     �l@     �n@     �j@     �p@      l@      o@     �q@     q@     �r@     �r@     �s@      u@     pu@     �t@     Pu@     x@     �w@     py@     �|@     �~@     �{@     �~@     �@     @@     ��@     x�@     ��@     `�@     ؄@     ��@     ��@     ��@     Ȋ@     �@     P�@     X�@     Ԑ@     `�@     \�@     ��@     �@     ��@     |�@     �@     H�@     |�@     >�@     �@     ��@     @�@     �@     �@     ��@     P�@     ��@     ��@     ��@     _�@     ��@     ~�@     3�@     ɹ@     ��@     ӽ@     W�@     ��@     0�@     ��@     ��@     ��@    ���@    �W�@     �@    ���@    �(�@    ���@    �Z�@    ���@     p�@     ��@    ���@    ���@     r�@    @��@    @�@    ���@    @�@    ���@    �1�@    @��@     ��@    ��@    `��@     \�@    `��@    ��@    ��@    ���@    ���@    @��@     t�@    �$�@    �$�@    `��@    @;�@    �u�@    �C�@    �s�@    ���@    ���@    @��@    ���@     >�@    ���@    ���@    @_�@    @��@    �z�@    �B�@     �@    ���@    �i�@     J�@    @,�@    �)�@     ��@     ��@    ���@     G�@    @��@     x�@    �1�@     �@     �@     *�@     ��@     ؠ@     @�@     ؐ@     ȃ@     `z@     �l@      `@      @        
�
predictions*�	   `_�   ���@     ί@!  `��/�)3k����4@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���[���FF�G �X$�z�>.��fc��>O�ʗ��>>�?�s��>x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?3?��|�?�E̟���?yL�����?�6v��@h�5�@�������:�              �?      @              @      "@      4@      F@     �P@     �O@      T@     �X@     �W@     �Z@     @^@      Z@     �a@     �a@     @[@     �\@     @Z@     �Z@     �R@      T@     �X@      R@      S@      N@      I@     �I@     �K@     �C@      C@     �B@      @@      C@      8@      1@      2@      2@      3@      3@      .@      (@      @       @       @      @      @      @      .@      @      @      @       @      @       @      @      �?               @      @       @              �?       @      @       @              �?               @      �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?      @              �?       @      �?       @       @       @               @      �?       @      @       @      @      @      @      @      @       @      @      "@      *@      @      $@      (@      $@       @      *@      *@      2@      3@      .@      0@      0@      7@      8@      3@      =@      3@      >@      ;@      <@      K@      G@     �C@      J@      D@     �C@      G@      E@      D@      F@     �F@     �I@     �A@      D@     �B@     �F@     �@@      C@      <@      A@      9@      9@      6@      2@      0@      *@      2@      $@      (@      $@      @      @      &@      @       @      @      @       @      �?      �?               @       @      �?              �?              �?       @              �?      �?              �?        _J��"2      ��/�	Q������A*�d

mean squared error��3=

	r-squared@D�=
�L
states*�L	   @���   ��@    ��NA!�lZ�!�@)��N(�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              (@      e@     �l@     z@      �@     ��@     ��@     (�@     ̟@     ��@     e�@    ���@    ���@     ��@    �z�@     ��@    ���@    ���@    ���@     ��@     ��@     ��@     ��@    ���@    ���@    @��@     ��@    �t�@     ~�@    @��@    ���@     z�@    �/�@     +�@    ���@    @��@    @"�@    @C�@     ��@     ��@    ���@    �P�@    �l�@    ���@    ���@    �F�@    ���@     ��@    `L�@    `��@    `f�@    ��@    ���@    �i�@    ���@    ��@    `�@    @��@    �P�@    �p�@    � �@    @��@    @&�@     ��@    @�@    ���@    @i�@    @U�@     q�@    �h�@    � �@    ��@     F�@    ���@    ���@    ���@    �D�@    �f�@    ���@     Ŀ@     ��@     �@     c�@     ��@     �@     ��@     ��@     ��@     ��@     >�@     ��@     j�@     �@     �@     ��@     ҡ@     
�@     ��@     ��@      �@     ؗ@     ��@     ��@     ̔@     x�@     \�@     ��@     �@     P�@     ȋ@     ��@     H�@     ��@     H�@     �@     x�@     x�@     `�@     0�@     Ѐ@     �@     �~@     0}@      |@     @w@     �z@      x@     �u@     �v@     u@     `t@     Pr@     @s@      q@     pp@     �r@     Pp@     �n@      n@     p@     �l@      h@     �h@     �h@     `g@     �d@     @f@      d@     `g@     �c@     �b@     �c@     @c@     �c@     �`@      _@     @\@     �a@     @^@     @\@     �_@      `@      Y@      W@     @X@     �[@     �\@     �S@     �X@     �W@     @V@      X@     �U@      R@      R@     �R@     �T@      P@      P@      R@      Q@     @Q@     �I@      O@     �P@      R@      O@     @Q@      J@      K@     �J@     �I@      H@      E@      I@     �F@     �E@      H@     �C@     �H@     �C@     �F@      H@      <@      =@      F@      >@     �@@     �B@      A@      :@     �C@      @@      ?@      A@      <@      A@      3@      ;@      1@      8@      >@      7@      5@      <@      5@      .@      8@      8@      4@      5@      6@      *@      *@      3@      ,@      1@      0@      .@      .@      0@      ,@      1@      2@       @       @      (@       @       @      &@       @       @      @      "@      &@      *@      @      "@      1@      ,@      "@      "@      @      @      @      (@      $@      "@       @      "@      @      "@      &@      @      @      @       @       @      @      @      @      @       @       @      @      @       @      @      �?      �?      @      @      @      @      @      @      @      @      @       @              @     `k@     �h@       @      @       @       @      @      @      @       @      @      �?       @       @      @      @      @       @      @      @      @      @      @      @      @      @       @      @      @      @      @      @      (@      �?      @      @       @      @      @       @      @      @      @      @      @      @      *@      @      @       @      "@      @      (@      $@      &@      $@      (@      @      "@      1@      .@      (@      *@      &@      *@      5@      *@      1@      .@      5@      ,@      0@      7@      ,@      6@      1@      1@      7@      3@      6@      8@      .@      5@      3@     �A@      6@      5@      =@      =@      6@      8@      C@      9@      >@      ?@      B@      >@      @@      ?@      9@      C@     �B@      ?@      E@      C@      A@      C@      H@     �B@     �E@     �B@      F@      E@      N@      H@      K@     �J@     �O@     �O@     �H@      Q@      O@     �Q@     �O@     @R@     �W@      T@     @Q@      T@     @T@     �V@     @U@     �X@     �V@     @Y@     @W@     @[@     �Y@     �]@      \@      [@     �^@      \@     @`@     �`@     �d@      b@     �b@      b@      a@     �c@      d@     �d@     `h@     �d@     �d@     �e@      i@     @l@      l@     �j@     �n@     @m@     @p@     �m@     p@     0q@     `q@     `r@     0s@     �s@     �u@     `v@     Pv@     `z@     �v@     �w@     0y@     �y@     �z@     �~@     �@     �@     ��@     (�@     ��@     ��@     ��@     �@     ��@     ��@     `�@     0�@     0�@     ؏@     ��@     ��@     �@     L�@     ̔@     ̔@     d�@     L�@     �@     Ԝ@     �@     >�@     ��@     ��@     �@     p�@     ��@     ҩ@     <�@     ^�@     N�@     U�@     ^�@     ��@     ��@     <�@     �@      �@     �@    ���@    ���@     ��@    �i�@    ���@    ���@     ��@    �A�@     -�@     ��@     �@    @��@    ��@    ���@    @<�@     d�@    ���@    ���@    ���@     ��@    ��@    ���@     ��@    �D�@    �
�@    ���@     ��@    �&�@    @��@    ��@    `n�@    @��@     ��@    @B�@     ^�@    ���@    @=�@    �)�@     ��@     ��@     ��@    �j�@    �P�@    �:�@     i�@     ��@     ��@     �@    �~�@     �@     o�@    ���@    ��@    ���@    ��@     H�@     ��@    @]�@    �<�@    �H�@    �C�@     ��@    @��@     K�@    �4�@    @��@    ���@    �]�@     s�@     n�@     ��@     �@     $�@     ܚ@     Ԓ@     ��@     ��@     �z@     �h@      $@        
�
predictions*�	   `�ȿ   �A��?     ί@!  P�i�"�)H�0f� @2��@�"�ɿ�QK|:ǿ!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.��f�ʜ�7
�������FF�G �>�?�s���;�"�qʾ
�/eq
Ⱦpz�w�7�>I��P=�>��[�?1��a˲?>h�'�?x?�x�?��d�r?�5�i}1?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?cI���?�P�1���?�������:�              �?              �?       @              @       @      &@      0@     �@@      I@      P@     �T@     @V@     �T@     �U@     @W@     �Y@      Y@     �Z@      Y@     �\@     @Y@      Z@     @W@     �W@      T@     �T@     �Q@      L@     �O@     �P@      C@      <@      E@      E@     �@@      :@      7@      3@      4@      *@      4@      .@      4@      5@      3@      3@      &@      ,@      *@      @       @      @      @      @      @      @      @      @      @      @      �?      @      @       @       @       @      �?      �?      �?       @      �?      @       @      �?               @              �?              �?               @              �?              �?               @               @      �?      �?       @               @      �?      @               @      @      @      @      @      @       @       @      @       @      &@      "@      &@      ,@      *@      .@      *@      ,@      $@      (@      4@      6@     �A@      <@      B@      4@      @@      =@     �B@      E@     �F@      J@      H@      O@      O@     �D@     �L@      M@      M@     �N@      I@     �F@     �K@     �B@     �E@     �E@      A@      A@     �@@      4@      6@      1@      *@      1@      ,@      &@      .@       @       @      @      @      @      @      @       @               @       @      �?       @      �?      �?       @      �?              �?      �?      �?              �?        �91r�2      �	6Y����A*�e

mean squared error�#=

	r-squared`�(>
�L
states*�L	   @���   ���@    ��NA!����@)�r*�6�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @      k@     Pz@     �{@     �@     h�@     ��@     @�@     �@     Z�@     ��@    ���@    ���@    �%�@    @W�@    � �@     "�@    ���@    ��@    ��@     )�@    ��@    �V�@    �,�@     ��@    ���@    �F�@    �x�@    @�@    @�@    @��@    ���@    �o�@    ���@    ���@     l�@    @��@    @�@    �m�@    @�@    �5�@    ���@    `��@    ���@    @]�@    ���@    �I�@    �%�@    ���@    ���@    �"�@    ���@     Y�@    @(�@    ���@    `��@    �z�@    ��@    ���@    ���@    ���@    �V�@     �@    ���@    �#�@    ���@    �*�@     ��@     ��@    @�@    ���@    ��@     s�@    �<�@    ���@    ���@     ��@    �0�@    ���@    ���@     �@     �@     ��@     ̷@     &�@     �@     ��@     [�@     ~�@     T�@     @�@     ��@     Z�@     �@     <�@     �@     "�@     �@     t�@     ��@     ��@     (�@     X�@     �@     �@     ��@     T�@     @�@     @�@     Ѝ@     p�@     ،@      �@     @�@     ��@     @�@     ��@     h�@      �@     ؀@     ��@      ~@     �}@     �{@     �{@     �w@     �x@     u@     �u@     �t@     `t@     Pt@     0q@     �s@     �r@      t@     �q@     @p@     0p@     �m@     @l@      l@     �j@     �k@     `g@     �h@      i@     `a@     �h@     �e@     �e@     �d@      d@     �d@     `b@     �d@      a@     �_@     �a@     �[@      Z@     �`@      a@     @[@     �a@     �\@     @Y@     �Z@      Z@     �Z@      W@     �W@     @W@     �W@      W@     @T@     �U@     �S@     �R@     @T@     �Q@     �O@      M@      O@     �Q@     �O@      O@      M@      O@      N@     �M@      K@     �J@      H@      K@      H@      B@     �K@      E@     �A@      D@      ;@     �@@     �D@      C@      A@     �G@      B@     �B@     �@@      <@      =@      ?@      @@     �B@      =@      5@      9@      4@      :@      7@      6@      ?@      2@      5@      *@      7@      <@      2@      :@      0@      (@      *@      *@      2@      .@      0@      (@      3@      1@      0@      (@      $@      .@      0@      &@      .@      $@      3@      (@      .@      &@      "@      @      @      ,@      @      &@      @      @      (@      "@      @      "@       @      "@      @      @      @       @      @      @      (@      @      @       @      @       @      @       @      @      @      @      @      "@      @      @      @      �?      @      @       @      @      �?       @       @      @      @      @      @       @      �?      l@     �l@      @      @      �?      @      @      @       @      �?       @      @      @      @      @       @      @      @       @       @      @      @      @       @      @      @      @      @       @      @      @      @      @       @      @       @      @      &@      &@      "@      @      $@       @       @       @      ,@      $@      (@      "@      "@      "@      @       @      .@      ,@       @      0@      *@      "@      "@      *@      2@      &@      2@      0@      6@      1@      ,@      @      1@      1@      3@       @      4@      3@      0@      5@      6@      4@      3@      4@      4@      9@      8@      <@      2@      5@      >@      7@      3@      @@      ?@      =@      ;@      ?@      D@      ?@      =@      9@     �@@      E@      @@     �A@     �F@      G@      I@     �B@     �F@      B@      G@     �E@     �G@      B@      F@     @P@      L@      P@      N@     �M@      K@      S@     �L@     �R@     @S@     �S@      Q@     �T@      Q@      R@     @X@     @[@     �W@     �Y@      W@      Y@     @\@      \@     @\@     @\@     �^@     �_@     @\@     `a@     �^@     �`@     �`@      c@     �c@      c@     �d@     �b@      c@     �c@     �g@      i@      i@     `g@      l@     �g@     `h@     �j@      l@     �l@     pp@     �m@     �p@     �p@     Pr@     @s@     �s@     �t@     �u@     r@     @v@     u@     0w@     �v@     Py@     �z@      ~@      }@     �}@     P~@      �@     ��@     ��@     ��@     `�@     ��@     (�@     ��@     P�@     ��@     ��@     ؎@     0�@     ��@     �@     ,�@     p�@     ��@     �@     ��@     d�@     ��@      �@     ��@     
�@     �@     t�@     ��@      �@     ^�@     �@     l�@     a�@     b�@     ��@     ��@     �@     ��@     �@     ؼ@     ֿ@     J�@     �@     �@    ���@     �@    ���@     ��@    �n�@    �s�@    ���@     ��@     ��@    @g�@    ��@     �@     ��@     �@    @��@    @��@     ��@    ���@    ���@    `c�@    ��@    �n�@     �@     X�@    @��@    @��@    ���@     5�@    ���@    �{�@    `|�@    `��@    �C�@    ���@     ��@    ���@    �v�@    ��@    @G�@     /�@    ��@    @6�@    �g�@    ���@    ���@    @ �@     V�@    �h�@    � �@    @��@    @y�@    �7�@    �?�@     ��@    �"�@    �1�@     �@    �U�@    ���@     ��@     ��@    ���@    @��@     ��@    @z�@     ��@     ��@     ��@     ��@     �@     ��@     ��@     І@     `~@     �|@     @p@      (@        
�
predictions*�	   @IR�   ���@     ί@!  �(! B@)$��Κ�;@2�+Se*8�\l�9�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�>h�'��f�ʜ�7
���[���FF�G �>�?�s���8K�ߝ�>�h���`�>6�]��?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?�E̟���?yL�����?u�rʭ�@�DK��@�������:�              �?              @      @      (@      1@     �@@      A@     �D@      J@      Q@     �R@     @S@      N@     �S@      V@     �Q@     @Q@     �R@      V@     �M@     �P@     �Q@     �N@      L@     �I@      C@      :@     �C@     �C@      C@      :@     �@@     �B@      :@      8@      6@      ,@      .@      ,@      ,@      1@      &@      $@      $@      &@      &@      &@      @       @      @      @      @      @      @      @      @      �?      �?              �?      �?      �?      @       @      �?      �?       @      �?              �?              �?              �?              �?      �?              �?              �?      �?      �?               @              �?      �?      �?       @      �?              �?      �?       @      @      �?      @      @      �?      �?      @               @      @      @      @      @      @      @      @      @      @      @      @      (@      "@      *@      5@      ,@      $@      0@      2@      4@      3@      7@      =@      B@      A@      G@      H@      K@      A@     �G@     �Q@     @S@      K@     �S@     �V@     @P@     �V@      S@     �W@     @V@     @Q@      U@     �U@      R@     �M@     �O@     �H@     �E@     �F@      ;@     �B@      A@     �@@      2@      4@      ,@      @      &@      "@      @      @      @      @      �?       @      �?      @      @       @      �?      �?              �?               @      �?              �?              �?              �?        {�Z�2      �m�	��J����A*�e

mean squared errormG'=

	r-squared��>
�L
states*�L	   �؈�   �,�@    ��NA!,�S�e.�@)�x���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @     �r@     �z@     @v@     �{@     �@     Đ@     �@     x�@     ԥ@     d�@    ���@     ��@     ��@    ���@    ���@    @e�@    �&�@    @��@     "�@    �[�@     d�@     ��@    �u�@    @c�@    @~�@    �]�@     ��@    �9�@    ��@    ���@    ���@     �@    ��@    @>�@    �I�@    @N�@    ���@    �Y�@    @��@    �,�@    `R�@    ���@    @}�@    `�@    �l�@     ��@    @��@    `��@    ���@    �"�@     ��@    �f�@    @�@    ���@     ��@    �F�@    ���@     �@    ���@    @G�@     ��@    ���@    @��@    �l�@     �@     ��@     :�@    �,�@    ���@     ��@    �S�@     ��@    ���@     ��@    �O�@    ���@     ��@     ��@     d�@     ��@     C�@     ��@     ��@     �@     ��@     ް@     2�@     Ȭ@     4�@     2�@     Z�@     2�@     ��@     �@     ��@     (�@     �@     T�@     0�@     �@     ��@     8�@     �@     �@     ��@     p�@     �@     Ȏ@     �@     @�@     ��@     ��@     �@     �@     ��@     ��@     8�@     ��@     ��@     �@     �@     �{@     0|@     @z@     �y@     �x@     Pw@     `v@     �u@     `t@     `s@     �q@     �s@     0q@     �q@      q@     �q@     �n@     �o@     �k@     �l@      l@     �l@     �h@     `h@     �h@     �f@     `g@      i@      d@     �d@      e@      a@      d@      c@      b@     �\@     @b@     �a@     �`@      _@     `a@     �`@     �[@     @Z@      ]@     �X@     �U@      Y@      Z@     �V@      V@      U@      V@      U@     �X@     �T@      L@     @T@      W@     �Q@     @R@      O@     @P@     �O@      R@     �N@     @Q@     �O@      K@     �L@      G@     �E@     �J@      H@     �N@      H@     �L@     �M@     �B@      G@      E@      A@     �E@     �H@      @@      B@     �B@      E@      B@      B@     �C@     �@@      >@      8@      @@     �B@      A@      @@      9@      =@      6@      7@      3@      8@      6@      9@      5@      4@      4@      3@      (@      ,@      &@      4@      3@      *@      ,@      .@      .@      .@      (@      &@      *@      *@      ,@      .@      @      (@      "@      .@      .@      1@      $@      *@      ,@      @      @      "@      $@      @       @      @      "@      @      @       @       @      @      @      &@      "@      @      @      @      @      $@      "@      @      @       @      @       @      @       @      "@       @       @      @      @      @      @       @       @      @      @      @      @      @      @      @     `q@     �l@      @      @      @      @      @       @      @      @      @      @      @      @      $@       @      @      @      @       @      @      $@      @      @       @      @       @      @      @      @      $@      @      $@      @      @      "@       @      @       @      @      @      $@       @      $@      $@      @      3@      &@      "@      &@      @      (@      "@      (@      $@      .@      (@      $@      "@      (@      *@      ,@      &@      .@      ,@       @      6@      6@      .@      .@      0@      9@      5@      2@      (@      7@      0@      2@      ,@      6@      8@      =@      :@      <@      7@      ;@      9@      <@      B@     �@@      ;@      <@      >@     �D@      7@      6@     �B@      D@      D@     �C@     �B@     �C@     �E@      H@     �J@     �E@      H@      H@     �G@     �C@     �E@      M@      J@      M@      O@     �S@     �N@     @R@     @S@      Q@     �N@     �R@     �R@      T@     �U@     �V@     �U@      U@     �U@      V@      Y@      [@      \@     �Y@      Z@     �Z@     �X@      Y@     @Y@     �[@     �_@     �`@     �^@     `b@      `@     �b@      e@     `a@     @c@     `g@     �e@     �d@     @g@      g@      g@     �f@     @j@     �g@     @k@     �k@     �n@      l@      l@     pp@     @l@     �m@     �r@     �p@     �s@     �s@     @u@     @t@     �s@     0u@     `v@     @y@     Pw@     @y@     �y@     �{@     �@     �~@     Ѐ@     ��@     h�@     @�@     ��@     ��@     ��@     `�@     І@     �@     �@      �@     ��@     h�@     l�@     (�@     <�@     ��@     ��@     (�@      �@     @�@      �@     4�@     ��@     j�@     ޡ@     ֢@     (�@     ��@     :�@     L�@     ��@     `�@     ��@     n�@     S�@     =�@     ��@     ��@     ~�@     �@     ׿@     ��@     6�@    ��@    ���@     <�@    ���@    �}�@     G�@    ��@    �b�@    ���@    �F�@    ���@    �~�@     \�@    @7�@     ��@    ���@    @��@    ��@     ��@    ���@    ��@    @��@    ���@    `B�@    @�@    �1�@    �i�@    `��@    ���@    �|�@    ���@    ���@    ���@    ���@     ��@    �5�@    `��@    ���@     �@    �c�@    @?�@    ���@    @��@    �@�@    �p�@     )�@    `�@    ���@    �2�@    ��@    ���@    ��@    ���@    �B�@    @��@    �N�@     ��@     ��@     ��@    �G�@     z�@    ���@    ���@     ��@    @��@     a�@     ��@     ��@     �@     ��@     ��@     �@      �@     `{@     Pu@     pw@      6@        
�
predictions*�	   ��i¿   �g�@     ί@!  H?�$�)���9@2��?>8s2ÿӖ8��s����(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��1��a˲���[��pz�w�7��})�l a���Zr[v�>O�ʗ��>��[�?1��a˲?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?������?�iZ�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�Š)U	@u�rʭ�@�������:�              �?               @      @      (@      5@      ?@     �E@      M@      P@      Z@     �V@     �Y@     �V@     @\@     �\@     �[@     �]@     @W@      W@     �W@      \@     @X@     �R@     �S@     @Q@     @S@     @R@      H@      P@      G@      J@      A@     �B@     �B@      ?@      4@      :@      ;@      4@      9@      ,@      ,@      "@      &@      @      @      (@      $@      @      @      @      @      @      @      @       @      @      @      @      @      @      �?      �?              �?      �?       @      �?      �?       @              �?              �?       @              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?       @      �?               @              @      @      @      @      @      �?       @      "@      @      @      @       @      @      @      @       @      "@      &@      @      *@      *@      1@      ,@      1@      5@      5@      5@      <@      <@      3@      @@      7@      ?@     �D@      E@     �K@     �D@      F@      H@     �F@      D@     �E@      K@     �I@     �I@     �H@     �I@     �J@      ;@      E@     �@@     �F@      9@      <@      6@      3@      ,@      &@      &@      1@      "@      $@      (@      @      "@      @      @      @      @      @       @      �?       @      �?      �?              �?              �?              �?              �?              �?              �?        �&�Բ2      ���	Ĕ�����A*�e

mean squared error�24=

	r-squared0M�=
�L
states*�L	   ���   ���@    ��NA!0�y�Gq�@)�iH�7A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@     `y@     �y@     `u@     �|@     ؇@     ��@     `�@     6�@     d�@     t�@    ���@     ��@    ���@    �6�@     ��@    �p�@     J�@    ���@     A�@    �N�@     3�@    ���@     ��@    @�@    @��@    ���@    @��@    ���@    @�@    ���@    ���@     �@    ���@    �a�@    @]�@     ��@    �l�@     {�@     k�@     ��@     ��@    `��@    ���@     [�@    ��@    ��@     �@     ��@    �]�@     �@    ���@     J�@    ���@    �t�@     -�@     ��@    `z�@    @��@    �O�@    @�@     ��@    @��@    ���@    @S�@    ���@    @��@     	�@    ���@    �%�@    ��@     n�@    ���@    �f�@     0�@    ���@    �E�@     ��@     �@     N�@     #�@     ú@     Q�@     	�@     �@     �@     $�@     <�@     ��@     j�@     h�@     Ψ@     ��@     $�@     ��@     *�@     J�@     l�@     `�@     �@     ؙ@     8�@     ��@     4�@     ��@     ��@     L�@     $�@     H�@     ��@     ��@     ��@     ��@     X�@     ��@     ��@     ؅@     �@     ��@     ��@     P�@     X�@     �}@     �{@     �{@     P}@     P|@     �{@     �x@     �w@     Pw@     0w@     @s@     0s@     0s@     �q@     �s@     r@     @o@      p@     �q@     `l@     @j@      h@     �h@     `k@      j@     `h@     �h@     �e@     �g@     �f@     �`@     @i@     �d@      a@     �`@     @c@     �[@     �b@      `@     �`@      `@      ]@     �X@     �\@     �\@     �Y@      \@      \@     @T@     �V@      T@     �W@      ]@      V@     �W@     �V@     �V@     �R@     �S@     @P@      R@      N@      S@      R@      Q@      O@     �P@     �P@     �G@     �L@     �H@     �N@      L@     �I@      I@     �H@      A@      I@      O@      G@     �E@      F@     �D@     �I@     �C@     �G@     �A@     �A@      ;@     �C@     �@@      4@      =@      <@     �C@      8@      9@      ?@      1@      ;@      6@      <@      ?@      1@      7@      8@      ;@      3@      0@      4@      0@      5@      5@      4@      A@      2@      2@      0@      ,@      3@      $@      (@      ;@      ,@      7@      3@      *@      "@       @      &@       @      @      "@      (@      @      @      .@      *@      &@      @       @       @      @      @      $@      @       @      @      @      @      @      @      @      @       @       @       @      @      @      @      "@       @       @      @       @      &@      @      @      @      @      @      @       @      @      @       @      @              @      @     `i@      k@      @      @      @       @      @      @      @      @      @      @      �?      @      @      @      @       @      @      $@      @      @      @      @      @       @       @      @      @       @      "@      @      @      "@       @      @      "@      $@       @      *@      @       @      $@       @      @      @      "@      ,@      $@      $@      $@      "@      (@      .@      2@      1@      .@      0@      3@      (@      @      .@      &@      4@      *@      ,@      5@      4@      5@      ;@      2@      ,@      3@      ,@      7@      5@      <@      9@      >@      ,@      8@      ?@      <@      4@      ;@      8@     �@@      A@      C@      ?@      @@     �D@     �B@     �E@     �E@     �G@      @@     �G@      E@     �C@     �G@     �I@      F@      A@      H@      L@      A@     �D@     �K@      L@     �I@     @P@      Q@     �M@     �O@      Q@     @R@     @R@      R@     �R@      R@     �S@     �T@      X@     @S@     �S@      V@     �X@      X@     �Z@     �W@      X@     �X@      Z@     @^@     �Z@     �_@     �]@      ]@     @[@      a@     @`@     `b@      a@     �d@      c@      e@     `b@     `c@     �c@     @g@     �g@     �d@      f@     �h@     �f@     �g@     `k@     @i@     @k@     �p@      m@     �o@     pp@     @p@      q@     �p@     0p@     t@     �s@     �v@     �u@     Pu@     `w@     �x@     �x@     �z@     @z@     �|@     }@     @@     �~@     P�@     �@     h�@     ��@     (�@     ��@     ��@     �@     ��@     �@     �@     �@     ��@     ܐ@     ��@     Б@     �@     X�@     �@     X�@     �@     p�@     H�@     ��@     ��@     \�@     �@     x�@     �@     ��@     ܨ@     L�@     ��@     ��@     d�@     :�@     ѳ@     ��@     ��@     ��@     ��@     ��@     H�@    ���@    �w�@    ���@     ��@     e�@    �G�@    �F�@     ��@    ��@     ��@     �@    ���@     ��@     ��@    ���@    ���@    @��@    �\�@    ��@     .�@     \�@    `	�@     ��@    �)�@    ���@    ���@    ���@    ���@     =�@    ���@    @$�@     {�@    `G�@    @,�@     :�@     ��@    ��@    ��@    ��@    `a�@     ��@    @)�@    @=�@    �z�@     ��@    @[�@     f�@    �5�@    ���@    ���@     {�@     
�@    @��@    @��@    �%�@     L�@    ���@    �T�@    �	�@    �p�@    �(�@    ���@     ��@    ��@    ���@    @�@    �X�@     1�@     k�@     t�@     ~�@     N�@     ��@     X�@     `�@      |@     `n@      y@     @R@        
�
predictions*�	    �ٿ   ����?     ί@!  D{,L@)z�k�׾3@2�W�i�bۿ�^��h�ؿ%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[���FF�G �>�?�s����h���`�8K�ߝ��*��ڽ>�[�=�k�>�f����>��(���>6�]��?����?�5�i}1?�T7��?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?ܔ�.�u�?�������:�              �?              �?              @      0@      ,@      9@     �A@      F@      F@     �H@     �O@     �R@     �N@     �R@      P@      L@     �L@      M@     �N@     �L@     �M@     �L@     �J@     �D@      G@     �A@      A@     �C@      D@      9@      <@      *@      1@      :@      7@      2@      3@      6@      5@      0@       @      @      "@       @      @      @      @      @       @      @      @      "@      @      @      @       @       @      @      @      @              �?      �?      �?       @      �?       @              �?      �?              �?       @              �?              �?              �?              �?              �?              �?               @              �?              �?               @      �?               @      @              �?               @      @       @       @      @      @      @       @      @       @      @      �?      @      @      @      @      &@       @       @      0@      *@      (@      7@     �A@      4@      5@      6@      >@      :@      <@      >@      =@     �D@      M@     �N@      K@     �S@      L@     @P@     �T@     �W@     �V@      T@     @Z@      Y@     @V@      W@      Y@     @Y@     �S@     �S@     @R@     @P@     �J@      L@      E@      C@     �D@      0@      9@      7@      ,@      &@      ,@      @      &@      @       @      @      @       @      �?      @      @       @       @      �?      @      �?      �?              �?              �?              �?      �?        Q�OO�2      u�	��
����A *�e

mean squared error�!"=

	r-squaredz1>
�L
states*�L	   `���   `~�@    ��NA!�>D��_�@)�E��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              4@     P}@     0v@     w@     �@     ��@     �@     ��@     ΢@     T�@     ��@    ���@    @�@     ��@    ���@    �H�@    �w�@    �C�@    �~�@     ��@     H�@    �D�@      �@     ��@     �@    ���@     	�@    ���@     ��@    ���@    ���@    @(�@     m�@     C�@    �"�@    @��@     �@    ���@    @��@    @��@    ���@    ���@     ��@     ��@    ���@    ���@    �(�@    ��@     ��@    @��@    @L�@    ���@    �p�@    ���@    @��@    �b�@    �%�@    @��@    ��@    ��@    �i�@    @^�@    ��@    ���@     ��@    �r�@     ��@     X�@    @��@    @��@     J�@     7�@    �`�@     ��@     ��@     x�@    ���@    �H�@     ��@     ��@     a�@     b�@     ��@     p�@     ��@     b�@     Ӱ@     ��@     ά@     ��@     X�@     ��@     n�@     ֣@     ��@      �@     ��@     t�@     t�@     ��@     ��@     �@     ��@     0�@     @�@     ȑ@     4�@     `�@     @�@     Њ@     ��@     ��@     ��@      �@     �@     ��@      �@     ��@     ��@      �@      �@     �~@     �@     0~@     �{@     �z@     @y@     �y@     �w@     �w@     �w@      u@     �s@     �t@     q@      q@     �n@     �p@     �p@     �m@     @o@     @o@     �k@     @k@     @i@     @i@     �i@     �f@      h@     �g@      f@     �c@     �f@     @f@     @e@     �b@      a@     �`@     �c@     �`@     �a@     �b@     @`@     @]@     @\@     @^@     �_@      \@      ^@     �Y@     �W@     �Y@     @T@      T@     �Z@     �V@     �V@      U@     @R@     �S@     @T@     �P@     �Q@      Q@     @T@     �S@     @R@      P@      I@      R@     �K@      M@      N@      E@     �I@      K@      F@      K@      K@      P@     �I@     �H@      B@     �C@     �@@      A@      D@      D@      B@     �E@      C@     �C@      B@      :@      :@      9@     �@@      A@      ?@      A@      >@      A@      ?@      0@     �@@      =@      <@      3@      5@      =@      6@      7@      ,@      .@      .@      :@      5@      &@      4@      1@      *@      *@      3@      0@       @      ,@       @      "@      "@      @      $@      ,@      (@      @      2@      &@      (@      .@      ,@      ,@      $@      ,@      @       @      *@      0@       @      @      "@       @       @      @      @      "@      (@      @      @       @      @      �?      @      @      @      @      @      @      (@      @      @      @      @      �?      @      @      @      @      @       @      @       @      @       @     @n@     �o@      @       @      @      @       @       @      @      @       @       @      �?      @       @      @       @      @      @      @      �?       @      $@      @       @      @      @       @      @      &@       @      @      @      "@      @      &@       @       @      "@      @      $@       @      @      1@      @      @      2@      $@      &@      &@      (@      $@      (@      (@      4@      @      0@      4@      3@      5@      ,@      2@      2@      0@      5@      3@      0@      ,@      7@      7@      7@      5@      ?@      5@      4@      5@      0@      =@      :@      9@     �B@      5@      >@     �C@      ;@      @@      ?@     �@@      A@     �G@      :@      6@      =@     �A@      A@     �C@     �E@      E@      H@     �D@      A@     �E@     �J@      H@      I@      L@     �K@     �F@      K@     �G@     �M@     �H@      I@      M@      N@     �P@     �R@     �T@     �S@      S@     �T@     �S@     �R@     �R@     �O@     �U@     @\@     �X@     @Y@      V@     �]@     @[@      Z@     @[@     �]@     @_@      Y@     �]@      ]@     �Z@     @`@     �a@     �b@     `b@     ``@     �d@     �e@     �d@     �c@      c@     `h@     �b@      e@      j@      j@     �j@     �i@     �l@      n@     �k@      n@     @p@     Pq@     `n@     �p@     Pq@     �q@     `r@     �s@     �s@     �t@     �q@     `u@     `w@     �w@     �w@     y@     `y@     p|@     P}@     @@     @�@     ��@      �@     ��@     ؂@     �@     ؃@     ��@     p�@     X�@     H�@     ��@     x�@     ��@     �@     ��@     X�@     ��@     t�@     @�@     ,�@     ��@     ��@     ��@     �@     `�@     x�@     ��@     l�@     ��@      �@     z�@     |�@     �@     ��@     #�@     ��@     �@     ��@     ��@     I�@     �@     [�@     9�@    ���@    ��@    ��@    �M�@    �`�@     z�@     n�@    �,�@     �@     ��@    @#�@     k�@    @K�@    @U�@    ��@    @0�@    ���@    @��@    ���@    ���@    �:�@    �6�@    ��@    `��@    ���@    `#�@     �@    `��@     ��@    �D�@    @��@    ���@    ��@    @��@     ��@    `�@    ��@    ���@    @��@    ���@    ���@    @��@    @��@    `��@    �I�@    ���@     �@    `��@     �@     ��@     c�@     B�@    @��@    @��@     C�@     ,�@     ��@    �8�@    @��@    � �@     ~�@     ]�@     g�@    ���@    �A�@    ���@     b�@     K�@     5�@     �@     ��@     ��@     ��@     ��@     ��@      |@     0q@     �v@     �a@        
�
predictions*�	   �=��   ��4@     ί@!  @�+�@)�
��c9@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�x?�x��>h�'��1��a˲���[���ߊ4F��h���`�
�/eq
Ⱦ����ž��~���>�XQ��>�f����>��(���>�h���`�>�ߊ4F��>����?f�ʜ�7
?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?u�rʭ�@�DK��@�������:�              @       @      0@      8@      >@      B@     �M@      O@     �L@     �P@     �V@     @V@      U@     @W@     �V@     �S@     �S@     @W@     @Y@     @V@     �U@     �P@     @Q@      P@      K@      P@     �N@     �O@     �E@     �I@      C@      ;@      7@      8@      9@      4@      6@      1@      0@      ,@      .@      2@      *@      "@      @      @      "@      @      $@       @      @      @      @      @      @      @      @               @       @      @      �?      @       @      @      @              �?      �?      �?      @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @       @      �?              @              @              @       @       @      �?      @       @      @      @      @      �?      @       @       @      @      ,@      @      *@      .@      (@      ,@      4@      5@      (@      8@      2@      4@      7@      A@      A@      ;@      C@      A@     �C@     �G@     �I@     �J@      N@      J@      L@     �Q@     @P@     @R@      N@     �L@      M@      K@     �M@      H@     �G@      J@     �@@     �E@      D@      B@      6@      1@      ;@      4@      5@      &@      8@      &@      (@      @      @      �?      @      @      @      @       @       @              �?               @              �?              �?              �?               @              �?        6�y|23      ��	x�f����A!*�f

mean squared errorsI'=

	r-squared�>
�L
states*�L	   ����    �@    ��NA!�O�=!�@)�iӮ�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              B@     �~@     �u@     0x@     ��@     X�@     (�@     d�@     
�@     2�@     �@    ���@    ���@     /�@     ��@    ���@    @��@    @��@    �^�@    ���@    ���@     �@     ��@     w�@     �@    �Z�@     ��@    �b�@    @K�@     ��@    @�@    ���@    ��@    ��@    �h�@    @��@    ���@    ��@    �9�@    @I�@    `�@    ���@    ���@      �@     ��@    `��@     ��@    `b�@    �t�@    `��@    ��@    �\�@    `��@     ��@     �@    ���@    �~�@     �@    ���@    @ �@    @P�@    @��@    @!�@    ���@    ���@    �c�@    ��@     �@    @�@     ��@     ��@    ���@     �@    �`�@    ���@     ��@     ��@    �f�@    ��@     I�@     ׺@     C�@     ~�@     ��@     �@     ��@     b�@     N�@     �@     4�@     Z�@     �@     �@     T�@     ԡ@      �@     ȟ@     ��@     ��@     <�@     P�@     `�@     ؕ@     T�@     ܒ@     t�@     ��@     �@      �@      �@     x�@     �@     @�@     ��@     ��@     ��@     ؅@     ��@     ��@     P�@     0�@     X�@     0~@     �~@     {@     `{@     �w@     �x@     �w@     �x@     0u@     �w@     @u@     Pt@     �s@     �t@     `s@     �q@     �p@     �q@     �p@      p@     �m@     @k@     @l@     �k@     @k@     �j@     �e@     @g@     �g@     �d@     @e@     �e@     �b@     �b@     @c@     `e@     �b@     `c@     �a@     �_@     �\@      a@     �a@      a@     @Y@     @`@      Z@     @\@     �\@     @W@     �W@     @Y@     �X@     �X@     @T@     �T@      Y@     �U@     �R@      T@     �T@     �V@      N@     �P@      S@      S@     �R@     @S@      S@     �K@      P@      M@     @P@      I@     �O@      L@     �G@     �L@      D@      N@     �D@     �G@      J@     �F@      @@      A@     �@@     �D@     �C@      E@     �E@      G@      E@      A@     �A@      :@      >@      ?@     �A@      3@     �B@      9@      =@      6@      <@     �E@      7@      <@      6@      7@      6@      9@      1@      4@      ,@      2@      8@      2@      ,@      .@      ,@      .@      6@      $@      .@      2@      6@      (@      $@      2@      2@      *@      (@      @      0@      $@       @      *@      1@      $@       @      $@      .@      $@      @      (@      .@       @      @      &@      &@      ,@       @       @      "@      *@      @      @       @      "@      @      &@      &@      @      &@      @      &@       @      @      @      @      @      @       @      @      @      @       @      �?      @     �t@     0s@      @      @      @      @      @      @       @      @      @      @       @      @      �?      @      �?      @      &@      @      @      @      @      @       @      @      $@      @      (@       @      $@      &@      @      @      *@      &@      $@      @      @      @      ,@      *@      &@      "@      *@      ,@      2@      (@      1@      0@      2@      1@      1@      *@      2@      :@      ,@      ,@      3@      5@      0@      :@      6@      7@      0@      1@      3@      .@      3@      2@      4@      :@      7@      >@      =@      4@      8@      @@      >@      <@     �A@      >@      8@      A@      ?@      D@     �C@     �A@     �B@      H@      H@     �E@      E@      @@      A@      G@      I@      K@     �J@     �G@     �H@     �E@     �L@      F@     �I@     �G@      N@     �J@     �H@      M@     @P@      J@     �Q@      T@     @Q@      Q@     @S@     �R@      T@     �P@     �R@     @Q@     �R@     �R@     @T@     @V@     �Z@      Z@     @]@     �Y@     @Z@     �Z@     �W@     �Z@     @\@     @]@     @]@     �^@     �`@     `a@     �b@     �d@      _@     @c@     @c@     �c@      c@     @c@      f@     �d@     @e@     �e@      h@     �g@     �h@      j@     `n@     �g@     �i@     �n@     �n@     �q@     `p@     pp@     �q@     �r@     �q@     s@     �t@     �s@     Pr@     �w@     0v@     �u@     px@     px@     �x@     z@     �|@      z@     X�@     P�@     ��@     p�@     ��@     x�@     X�@     ؄@     p�@     ؇@     ��@     ��@     ��@     ��@     ��@     d�@     `�@     ��@     p�@     ��@     ��@     ܕ@     �@     �@     h�@     �@     L�@     �@     ء@     ��@     Ĥ@     ^�@     �@     ��@     ��@     ��@     /�@     �@     1�@     l�@     �@     t�@     G�@     ��@     T�@     ��@    ���@    �X�@    �?�@    ���@    �y�@     ��@     �@    �j�@    @�@     \�@    ��@    @��@    �Z�@     ��@    ���@     ��@     ��@    �n�@    ���@     ��@    ���@     ��@    `��@     �@    @��@    @��@     &�@    ��@    ��@    `��@    ��@     ��@     ��@    ��@     ��@    �8�@    ��@    ��@    ���@     ��@     ��@    @y�@     ��@    �F�@     ��@    �n�@    �.�@    �s�@    �5�@    ���@    ���@    � �@     (�@     ��@     ��@    @.�@    @��@    ���@    ��@    �w�@     �@     ��@    @��@    ���@    @��@    @��@    ���@     �@     �@     �@     �@     �@     �@     ��@     0z@      l@     �s@      i@        
�
predictions*�	   @J���    ܿ@     ί@!  p���@)�eZWy�7@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��f�ʜ�7
������6�]���1��a˲���[��O�ʗ�����Zr[v��K+�E���>jqs&\��>���%�>�uE����>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?3?��|�?�E̟���?h�5�@�Š)U	@�������:�               @       @      @      *@      <@      @@     �L@     �P@     �Q@      Q@     @T@     @S@     �Z@     �W@     �[@     �[@     �[@     �`@     �X@      ]@      U@      S@     �U@     �M@     @Q@      G@     �J@     �A@      B@      C@      D@     �B@      <@      B@      6@      >@      8@      *@      *@      2@      1@      0@      @      (@      *@      @       @      @       @      @      @      @      @      @      @       @      �?       @      @      @      @      @      @      �?       @      �?      @       @               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?               @               @      �?              �?      �?               @              �?               @       @      @      @      @       @       @      �?      @      @      @      @      "@      @      @      @      &@      "@      @      @      @      *@      7@      3@      3@      $@      4@      6@      6@      ?@      ;@      >@      C@     �E@     �B@     �E@     �K@      G@     �J@      M@      E@     �F@     �G@     �H@      J@      L@     @P@     �K@      F@     �B@      G@     �E@     �C@      B@      9@      D@      ;@     �A@      5@      .@      :@      (@      *@      .@      "@      @      @      @              @      @              @       @       @              �?              �?      �?      @               @              �?        Z6Ѿ3      ��<	3�Ë���A"*�f

mean squared error�$=

	r-squared\%>
�L
states*�L	   @{��    L�@    ��NA!�)"���@)�v �
A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              G@     `}@     �x@     ~@     ��@     ��@     �@     �@     ��@     �@     ��@     ��@    @��@     O�@    @��@     z�@    @��@    @��@     ��@     ��@     i�@     �@    �!�@     ��@    ���@    @@�@    @��@    ���@     ��@    ���@    ���@    @��@     8�@    �2�@     ��@     &�@    @�@    ��@    ��@    ���@    @b�@    ���@    ���@    �^�@    ���@     ��@    ���@    ���@    ��@    ��@    @��@    @�@    �v�@    ���@    @��@    �~�@    �T�@    `��@     ~�@    ��@     ��@    �&�@    @��@     �@     l�@    @�@    ���@    ��@    @��@    �x�@    �D�@     ��@    �y�@     ��@     6�@     ��@    ���@     *�@    ���@     j�@     ]�@     ��@     �@     O�@     �@     y�@     ��@     3�@     ��@     4�@     h�@     ��@     D�@     ��@     ��@     �@     L�@     `�@     ��@     |�@     ̘@     ��@     Е@     `�@     Г@     ��@     x�@     �@     �@     (�@     0�@     8�@     (�@     `�@     x�@     ��@     ؄@     `�@     @�@     ��@      �@     X�@     @�@     �~@     �~@     �|@      {@     �z@     �v@     Pw@     �w@      x@     0u@     �s@     ps@     �r@     �t@     �q@      q@     @r@     �q@     `n@     �n@     `o@     �j@     @n@     �k@     �l@      h@     �j@      g@     �k@     �d@     `e@     @g@     �g@     `a@     �`@     `b@     �c@     �[@     @`@     @a@     @`@     �a@     �\@     @^@     ``@     �`@     �[@     �[@      [@     @\@      Z@     �\@     @Y@     @X@      V@      Y@      Y@      W@     @X@     �Q@     �V@     �W@      W@     �S@      K@     �R@      S@      R@      T@      O@     �P@     �P@     �P@     �R@      M@     �H@      H@      G@     �P@      J@      I@      G@      D@      D@     �D@     �A@      B@     �E@      F@      C@      >@     �@@     �D@     �C@     �F@      =@      @@      B@      ;@     �@@      ?@      :@      ;@      :@      6@      9@      0@      8@      7@      <@      6@      8@      1@      =@      4@      4@      4@      5@      2@      :@      .@      8@      &@      $@      5@      1@      0@      0@      2@      *@      3@      &@      $@      .@      .@      .@      3@      $@      .@      $@      @      (@      (@      $@      @      "@      *@      @      *@      @      "@      @       @      @      @      @      @      @      @      @      @      "@      @       @      @      @      @      @       @      @      "@       @      *@      @       @      @       @       @      @     �p@      r@      @      @      @      "@       @       @      @      @      @      @       @      @       @      @              @      @      @      @      @      "@      @      @      $@      *@      @      @      &@      @      $@      "@       @      @      "@      "@      @      0@      @      .@      ,@      .@      0@      .@      .@      "@      0@      0@      .@      "@      ,@      ,@      *@      4@      "@      1@      1@      ,@      3@      7@      5@      ;@      .@      5@      7@      7@      :@      :@      9@      8@      6@      ;@      ?@      :@      ;@      8@      9@      ?@      @@     �A@     �@@      8@     �C@     �A@      E@      :@     �A@      D@     �@@      D@      B@      A@      J@      E@     �J@     �K@      G@     �F@     �H@      M@      B@     �E@     �J@     �K@      G@      P@     @P@     �M@     �Q@     �Q@     @T@      P@      K@     �Q@     �R@     �S@     �P@     @R@      N@     �R@      S@     @T@      W@     �S@     �T@      Z@     @S@     @Z@      [@     �]@      `@     �^@     �`@      ^@     �Z@     @Z@      b@     �`@     �b@     @c@      c@     �c@     `c@     ``@     `a@     �g@     �b@      f@     �h@     `i@     �e@      f@     �m@     @l@     `h@     @o@     �i@     �l@     �k@     �m@     @p@     �q@      p@     `p@     �p@     t@     `u@     �q@     pu@     u@      v@     u@      y@     @z@     �y@     �y@     �y@     z@      ~@     `}@     ��@     �@     ��@     (�@     ��@     `�@     ��@     �@     H�@     ��@     x�@     ��@     x�@     ��@     @�@     0�@     Ē@     �@     ��@     ��@     �@     ��@     �@     �@     �@     �@     ��@     �@     |�@     ��@     Ħ@     �@     ��@     v�@     �@     ��@     ��@     ۲@     ��@     [�@     ��@     M�@     �@     ̾@    ���@    ���@     ��@     ~�@    ��@     k�@    �Y�@    �z�@    �"�@     l�@    @��@    @��@    �e�@    ��@    @	�@     -�@    ���@    @�@    @d�@    �c�@    �W�@    ��@     ��@    �+�@     k�@    `d�@    ���@     J�@    ��@     ��@     -�@    �i�@    ��@    ��@    ���@     �@     7�@    �#�@    �1�@    �5�@    ��@    �k�@    ���@    ���@    �L�@    �v�@    ���@    �#�@    �~�@     ��@    ���@    ���@    �m�@    �>�@    @h�@    ���@     9�@    @�@    @��@    �i�@    ���@    @��@    ���@    ���@     �@    �{�@    ��@    �^�@     �@     �@     ئ@     *�@     D�@     �@     P�@     @}@     �i@     ps@     q@        
�
predictions*�	   `";��   ��_@     ί@!  \1�$1@)�3�4@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��6�]���1��a˲��FF�G �>�?�s�����(��澢f����E��a�W�>�ѩ�-�>���%�>�uE����>I��P=�>��Zr[v�>O�ʗ��>1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?h�5�@�Š)U	@�������:�               @       @      @      &@      2@      :@     �D@     �C@     �J@      F@      O@     �T@     �V@     @P@     �T@      T@     �R@     �W@     �T@     �U@     �R@     �P@      M@     @Q@     @P@     �N@      M@      J@     �K@      ;@     �G@      9@      E@     �B@      8@      6@      :@      <@      6@      6@      3@      (@      2@      4@       @      (@      $@      "@       @       @      @      @      @      @      @      @      @       @      @      @      @      �?               @       @      �?       @              �?       @       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?       @               @              �?      �?              �?               @              �?       @      �?       @      @      �?      @       @      �?       @      @      @      @      @      �?      @      @       @      @      @      (@      *@      @      *@      (@      3@      4@      "@      4@      ;@      8@      ;@      4@      >@      @@      B@      >@     �A@     �E@     �C@      K@      M@     �Q@      N@     �R@     �L@      J@     @Q@     �Q@      P@     �Q@     �P@     @P@     �G@      G@      K@      I@     �E@     �F@      B@      C@     �E@      ?@      0@      4@      .@      *@      ,@      .@      (@      @      @      @      @      @      @      @      @              �?      �?      �?      �?      �?      �?              �?              �?      �?              �?        oɸ��2      ���C	�8����A#*�e

mean squared error7!%=

	r-squaredX�!>
�L
states*�L	   ���   ���@    ��NA!/�)���@)6os��]A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              D@      {@     �{@     `�@     ��@     H�@     ��@     (�@     �@     ��@     б@    ���@    @6�@     ��@    @?�@    @��@    ��@    @C�@    ���@     ��@     h�@     ��@    ���@    ���@    �0�@    ��@    ���@     �@     4�@    �m�@    @��@     g�@    ���@    @2�@    ���@     ��@     ��@    ���@    �$�@    ��@    ���@    ��@    ��@    ���@    ���@    �N�@     ��@    ���@     ��@    `h�@    ��@     C�@    ���@    `n�@    @�@    ���@    @_�@    ���@    �R�@    @X�@    @��@    @�@    ���@    ���@    �-�@    @��@    �.�@    @ �@     ��@     =�@     m�@    ���@     ��@    ���@     ��@    ��@    ���@     ��@     V�@     ̼@     R�@     ��@     ̶@     
�@     ��@     \�@     U�@     @�@     *�@     *�@     >�@     ,�@     ,�@     Ȥ@     ��@     T�@     ��@     ,�@     p�@     �@     ��@     ܗ@     �@     x�@     �@     ��@     ��@     đ@     �@     ��@     0�@     `�@     ؉@     `�@     (�@     �@     ��@     ��@     @�@     ��@     0�@     ȁ@     �@     (�@     @      {@     �{@     Py@      y@     �v@     �v@     �w@     pw@     pu@      v@     `t@     �t@     �q@     �s@      r@     @p@     �o@     �n@     �p@     �p@     `j@      m@     @k@     �h@     �i@     �l@      h@     �k@     @h@     �f@     �f@     @b@     @d@     �d@     �d@     `b@     @_@     �`@      `@     `a@     �`@     @a@      Y@     �`@     @\@     �[@     @^@     @Z@     �X@     �^@     �\@     �[@     �R@      X@     �W@      X@     �T@     �T@     @T@      T@     @S@     �T@     �U@     �R@     �O@     �P@      O@     �K@     @Q@      Q@      N@     �T@      O@     �L@     @Q@      J@      H@      L@     @P@     �K@     �L@      H@      @@      L@      D@      H@      I@     �H@     �A@     �F@     �F@      >@     �@@      D@      D@     �A@      =@      @@      G@      6@      9@      B@      >@      4@      7@      2@      7@      ?@      >@      .@      8@      ;@      ,@      ?@      9@      2@      8@      8@      1@      1@      7@      3@      4@      7@      ,@      5@      1@      6@      0@      5@      2@      0@      0@      *@      1@      "@      &@      *@      (@      &@      &@       @      &@      .@       @      $@      @      .@      @       @      "@       @      $@      "@      "@      @      @      $@      @      "@      @      @       @      @      �?               @      @      @      @      @      @      @      @      @      "@      @      s@     Ps@      @      @      @      @      @      @       @      @      @      @      $@      @      @       @      �?      @      @       @       @      @      @      @       @       @      "@      @      &@      "@      @      &@       @      "@      (@      (@      2@      &@      ,@      @       @      @      "@      (@      1@      ,@      2@      (@      &@      (@      .@      ,@      3@      .@      5@      5@      .@      0@      (@      2@      1@      4@      3@      0@      ;@      <@      5@      =@      ;@      @@      :@      =@      A@      <@      ;@     �B@     �@@     �D@      =@      ;@     �A@     �D@      B@      A@      =@      G@     �B@      D@      B@     �B@     �C@      F@      <@     �D@      G@      G@      G@     �C@      P@      I@     �P@     �G@      G@     �J@      P@     @Q@      K@     @Q@     @P@      P@      P@      R@     �I@      U@      O@      T@      V@     @T@     @T@     @X@     @V@     @T@     �[@     �[@     @Y@      Z@      U@     �V@     �^@     �Z@     �[@     �]@     @]@      ^@     �[@      `@     `b@      b@     �`@     �`@     @b@     �]@     �c@     �b@     �d@     @d@     �d@     �g@      f@     �e@     �e@     `g@     �i@     `h@      m@     `i@      i@      l@     �m@     �m@     p@     �n@      o@      q@     @s@     �q@     @q@     ps@     Ps@     �s@     �s@     �v@     �x@     �w@     �w@     �y@     Pz@     �{@     p|@     �}@     @~@     ��@     �@     `�@     ؂@     ��@     ��@     ��@     Ѕ@     0�@     ��@      �@      �@     �@     0�@     P�@     ��@     \�@     �@     ��@     ��@     $�@     ��@     �@     ��@     �@     ��@     |�@     ��@     2�@     �@     D�@     �@     ^�@     ��@     �@     x�@     Ű@     
�@     γ@     �@     ʶ@     c�@     ��@     f�@    �6�@     O�@    ���@    ���@    ���@     ��@    ��@    ���@     ��@    @��@    �#�@    ��@    @A�@    @/�@    ��@     ��@    �j�@    @��@    ` �@    �P�@    �6�@    �K�@     $�@    ���@    ���@     ��@    ��@    �x�@    @��@    �a�@    ���@     ��@    ���@    `��@    �"�@    ���@     '�@    ��@    �.�@    �;�@    @��@    ���@     ��@    `��@    �!�@    @��@     s�@    ���@    ���@    ���@     8�@     ��@     Y�@    ��@    @}�@     U�@    ���@    @!�@    �n�@    ���@    �.�@     ��@    ���@    �|�@     ��@    ���@    �p�@    ���@     x�@     �@     Ч@     b�@     �@     ��@     ��@     �~@     �p@     �s@     �r@        
�
predictions*�	    �Hֿ   ��@     ί@!  ��]w#�)<h���6@2���7�ֿ�Ca�G�Կ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��O�ʗ�����Zr[v��})�l a��ߊ4F��h���`�8K�ߝ뾢f�����uE�����u`P+d�>0�6�/n�>�XQ��>�����>��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?\l�9�?+Se*8�?������?�iZ�?u�rʭ�@�DK��@�������:�              �?              �?              @       @      "@      2@      @@     �@@     �D@     �N@     �Q@     �O@      R@      U@      Y@     �X@     �X@     �Z@     @V@     �Z@     @\@      X@     �Z@     �U@     @T@      W@     �J@     �R@      J@      K@      G@     �O@     �J@     �@@     �A@     �C@      =@      A@      ;@      "@      2@      :@      "@      1@      ,@      .@      $@      "@      "@       @       @       @      �?       @      @      @              @       @      @      @      �?      �?       @       @      @      �?       @      �?       @      �?       @       @               @               @              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @      �?      �?      �?      @       @       @               @      @      �?      @       @               @      @      @       @       @      @      "@      @      @      *@      &@       @      "@       @      (@      0@      (@       @      1@      3@      5@      3@      :@      ?@      6@     �A@      5@      >@      ?@     �C@      A@     �D@     �G@      I@     �H@     �K@     �I@      J@      F@     �A@      F@     �I@      A@     �B@     �C@      @@      D@      >@      D@      D@      A@     �@@      8@      @@      .@      .@      5@      &@      (@      ,@      0@      &@      @      @      @      @       @      @       @              �?      �?              �?              �?              �?        #��Sb3      ��v	�{�����A$*�f

mean squared erroru$0=

	r-squared��=
�L
states*�L	   �t��   ���@    ��NA!��/���@)v��:�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              B@     @|@     �{@     �@     ��@     ԓ@     p�@     ��@     ��@     ��@     ��@    ��@    @��@    ���@    ���@    ���@    ��@    @��@     ��@     ��@     �@     ��@     W�@     M�@    @�@    ���@    ���@     H�@    �T�@    @�@     ��@      �@    @;�@    �F�@    ���@    �-�@    ��@    �F�@    @��@    �Q�@    @f�@     h�@    �[�@     R�@    ���@    �f�@    ���@     ��@    ���@    �`�@    @�@    ���@     ��@     ��@    `b�@     "�@    ���@    �#�@    �r�@    ���@    ���@    @k�@    @x�@    �P�@    @��@    @c�@    ���@    �a�@    ���@     ��@     =�@    ���@    ���@     ��@     ��@     �@     ;�@    ��@     Ͼ@     ��@     ҹ@     S�@     ��@     ��@     ��@     �@     T�@     ��@     "�@     B�@     ڧ@     ʦ@     ĥ@     |�@     Ģ@     ��@     `�@     �@     l�@     Ț@     ��@     ��@     l�@     ܕ@     ��@     ȑ@     ��@     �@     ��@     ��@     ،@     H�@     ��@     (�@     ��@     ��@     ��@     h�@     h�@     �@     ��@      �@     �@     `}@     �|@     �|@     �|@     pz@     �y@     �y@     @y@     @v@      z@     w@     �u@     �r@      s@     �r@     `t@     �q@     �r@     `q@     �q@     �m@     �n@     p@      n@     @i@      h@     �l@      j@      j@      i@      h@     �h@     `g@     �d@     �d@     `c@     �c@     �b@     �d@     @c@      a@     �_@      ^@     ``@     �\@     @`@     �`@      _@     �]@     �^@     �]@     �V@     �X@     �[@     @V@     @Y@      Y@     �S@     �U@     �Y@     �T@     @T@     �S@     @X@      V@     �Q@     @U@     �T@     �R@     �S@      O@      Q@     �L@      O@     @P@      H@     �K@      F@     �P@      I@     �M@      I@     �F@      I@     �F@     �B@      H@      G@     �C@     �M@     �F@      D@      A@      @@      G@      D@      B@     �B@      A@     �@@      >@      ;@      =@     �@@     �D@      6@      :@      <@      6@      9@      9@      2@      :@      4@      :@      A@      2@      5@      5@      4@      8@      3@      9@      5@      7@      .@      9@      4@      ,@      9@      2@      $@      .@      0@      &@      ,@      *@      ,@      *@      1@      $@       @      &@      "@      "@      "@      $@       @      3@      *@       @       @      &@      "@       @      "@      $@      "@       @      @      @      @      @      @      "@      @      @      &@      @      @      @      @      @      @      $@      @      @      "@      @     �t@     �u@      @      @      @      @       @      @      "@      @      "@      @      "@       @       @      @      @      @      @      ,@      @      @       @      @      @      &@      @      "@      $@      &@      $@      0@      *@      @      "@      "@      @      @      @      .@      .@      .@      4@      1@      8@      ,@      1@      ,@      0@      3@      .@      7@      3@      *@      (@      0@      7@      6@      0@      3@      9@      .@      5@      9@      8@      0@      7@      =@      =@      4@      :@      =@      :@      ;@      8@      C@      =@     �B@      <@     �@@     �C@     �D@      <@      D@      C@      =@     �B@     �D@      ?@     �D@     �G@     �K@     �F@     �P@     �H@     �H@      H@     @P@      E@      L@      P@     �I@     �K@     @P@      N@      L@      K@     �L@     @Q@     @R@     �J@     @T@      U@     �P@     @U@     �O@      U@     �Y@     @R@      Y@     @U@     �W@     �X@     �T@     �U@     �U@     �Z@     @X@     @Y@     @`@      \@     @^@      [@     �`@     @\@     @_@     �_@     �b@     �b@     `a@      a@     �c@     �c@     `a@     �c@     �d@      d@     @d@      h@     @g@     �g@     @f@     �g@      h@     `j@     @m@     �n@     �m@     @m@      o@     @m@     �m@     pq@     0p@     �p@     �r@     Ps@     Ps@     Pt@     0s@     �u@     Pt@     0y@     x@     px@     �w@     Py@     �|@     �|@     @     �@     ��@      �@     ��@     @�@     ��@     �@     h�@     �@     �@     �@     ��@     x�@     ��@     �@     (�@     ��@     Б@     ��@     ��@      �@     0�@     D�@     ��@     Ԛ@     L�@     t�@     ��@     С@     �@     0�@     �@     �@     P�@     ��@     �@     �@     Ͱ@     C�@     �@     ��@     ��@     �@     �@     �@     ſ@     ��@     ��@     ��@    �K�@    �7�@    �d�@     ��@    �N�@    �.�@    @��@    @M�@    �Y�@    @�@    @��@    ���@    �|�@     '�@    �K�@    �u�@     ]�@    �0�@     ��@    �7�@    �f�@     ��@    ���@    �`�@    ���@    `��@     =�@    ���@    ���@    �|�@    @��@    `��@     ��@    ���@    ���@    �
�@    ��@    `=�@     ^�@    `��@    ���@    ��@    `��@    ���@    ���@    ���@    �D�@    @��@    @~�@    �{�@    ���@    ���@    ���@    @�@    @��@    �v�@    @��@    @��@    ���@    @��@    ���@    ���@    ���@    �Y�@     -�@     r�@     ��@     ̢@     �@     �@     ��@     @�@     �m@     @r@     �t@        
�
predictions*�	   `�1ҿ   ��#	@     ί@!  �'}nE@)�̌�E@@2�_&A�o�ҿ���ѿ�Z�_��ο�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�f�ʜ�7
������6�]�����[���FF�G �O�ʗ�����Zr[v��K+�E��Ͼ['�?�;�����>
�/eq
�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?h�5�@�Š)U	@�������:�              �?      �?               @       @      $@      @      0@      1@      1@      A@      L@      L@     �O@      Q@     @P@     �R@     @S@     �R@     @S@     �U@      U@     �V@     @Q@      U@      O@     �R@     �Q@      J@     �J@     �I@     �J@     �G@      G@     �F@      F@      I@      ;@      =@      6@      3@      5@      0@      &@      *@      0@      .@      (@      1@      &@      ,@      ,@      @      @      @      @      @      @      @      @       @      �?      @      �?      @      @      �?      @      @       @       @      @               @       @      @       @               @       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?       @      �?       @               @       @      �?      @              �?      @      @      �?      @      @       @      @      @      @      "@      @      @      ,@      $@      &@      *@      2@      2@     �@@      6@      9@      7@      7@     �@@      B@      =@      ?@      B@      C@     �A@     �G@      I@      Q@     �G@     @Q@      P@      O@      N@      J@     @P@      M@      N@     �H@     �K@     �J@      H@      F@      F@     �G@      @@      @@     �B@      @@      ;@     �B@      ;@      8@      5@      0@      $@      (@      "@      @      &@       @      @       @      @      @       @       @      �?              �?              �?               @      �?      �?              �?        #���2      {�T	XYߌ���A%*�e

mean squared errorm`#=

	r-squared��*>
�L
states*�L	   ����    ��@    ��NA!ۄ�	��@)G%A��0A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              E@     px@     �|@     0�@     H�@     L�@     p�@     N�@     �@     P�@     ��@    �v�@    ���@     N�@    �z�@    ��@    ��@    �h�@     ��@     ��@     ��@     �@    ���@    �
�@    �h�@    �\�@    �w�@    ���@    ���@    �9�@    `�@    ���@    �q�@    �!�@    �]�@    ���@    @��@    �6�@     �@    ���@     ��@    `��@    ��@     ��@    �C�@     ��@    ���@    ���@    @��@     I�@    ���@     Q�@    ���@    �n�@    ��@     ��@     J�@    @��@    ���@    �X�@    �
�@     C�@     ��@    ��@    ���@    ���@    �9�@    @��@    @|�@    �h�@     �@    ���@    �B�@    �h�@     ��@    ��@    ���@     |�@     �@     S�@     0�@     x�@     ��@     U�@     (�@     w�@     �@     ��@     ��@     2�@     ��@     ��@     ��@     p�@     >�@     ��@     �@     �@     ��@     ��@     ��@     ܖ@     p�@     8�@     В@      �@     `�@     X�@     ��@      �@     ��@     ��@     x�@     ��@     @�@     ȅ@     (�@     ��@      �@     `�@     ��@     Ё@     P�@     �@     h�@      @     P{@     �z@     `z@     Py@     @y@      w@     @v@     �w@     �u@     �t@     t@     �t@     �q@     �q@      r@     �p@     @o@     Pq@     p@      l@     �j@      o@     �k@     �j@     @g@      j@      i@     `j@     �h@     `c@      h@     �e@      b@     �c@     �d@     �d@     `a@     �b@     @b@     @^@     ``@     �\@      ^@      _@      a@     @[@     @\@     �_@     �Z@     �[@     �W@     �_@     �Z@     �X@     @S@      T@      X@      R@     �X@     �S@     @T@     �M@     �J@     @Q@      L@     �R@     �P@     �N@     �Q@     �R@      N@     �K@     �M@      P@     �E@      I@      I@      L@     �H@     �H@      F@      A@      D@     �E@     �F@     �G@      E@     �A@      K@     �I@      >@      >@     �C@      4@     �D@      9@      D@      F@     �B@      =@      A@      B@      <@      @@      ?@      4@      <@      4@      8@      2@      5@      3@      0@      4@      ;@      >@      5@      6@       @      1@      :@      *@      0@      &@      3@      *@      1@      .@      .@      $@      7@      8@      &@      $@      *@      (@      $@      &@      *@      *@      ,@      *@      "@      .@      @      @      @      "@      "@      "@      0@      &@      (@      "@      @      &@      @      @       @      &@      @      @      "@      @      @      �?      @      @       @       @      @      @      @      @      @      &@     @r@     pr@      @      @      "@      @       @       @      @       @      @      @      @      $@      @      @      &@      @      @      @      @      @      *@      @      (@      $@      @      "@      $@      *@      "@      &@      *@      $@      *@      (@      ,@      $@      *@      (@      *@      3@      1@      *@      0@      4@      "@      2@      &@      ,@      .@      ,@      7@      *@      1@      =@      :@      *@      6@      5@      5@      3@      6@      8@      4@      @@      7@      ;@      3@      @@      >@      B@     �@@      :@      <@      9@      :@      9@      ;@     �F@      :@      9@     �A@      D@      B@     �A@      B@      7@      >@      F@     �@@      L@     �D@      H@      D@      L@     �D@      K@     �G@     �H@      O@      N@      N@      M@     �Q@     �S@     �Q@      N@      Q@     �Q@     �R@     �L@     @Q@     �O@      O@     �R@     @T@     @R@      U@     �X@     @T@      V@     @T@     �T@     �\@     �X@     �Z@     �]@     �\@     �X@     �`@     �`@     �^@      _@     �`@     @^@     �\@      b@     `a@     �e@      a@     `d@      e@     @b@     `d@     @d@     �e@     @k@     �f@     �h@     �e@      i@     �j@     `k@     �l@     �i@      l@      n@      l@     �m@     �o@      q@     �r@     r@     �q@     @p@     @s@     Ps@     �t@     �t@     �v@      u@     �w@     Pw@     �z@     �x@     �{@     �|@     �}@     @@     @~@      �@     ��@     H�@     ��@     ��@     ��@     (�@     P�@     `�@     (�@     8�@     `�@     H�@     ��@     L�@     А@     h�@     ��@     `�@     ��@     �@     T�@     ̘@     \�@     ��@     \�@      �@     ��@     ��@     ��@     ��@      �@     ��@     ��@     ��@     F�@     "�@     ��@     D�@     ��@     �@     a�@     ?�@     ��@     �@     ��@     �@     ��@     ��@     ��@    �z�@    ���@    ���@     ��@    ���@    @��@    @�@    �W�@    �N�@    �Z�@     ��@     q�@    @��@     c�@     ��@     �@    ���@    �V�@    `_�@    `��@    ���@    ���@     )�@     ��@    ��@    ��@    ` �@    �o�@    ���@    ���@    ���@     ��@    `n�@     Z�@    ���@    ���@    ���@    ���@    ���@     ��@    ���@    @>�@    ���@    @��@    @��@     "�@    ��@    ���@    @��@    ���@    @��@     ��@    @,�@    @��@    @��@     ��@     ��@    ���@    ���@    ��@    @��@    ���@     ��@     D�@     Ī@     �@     $�@     |�@     ��@     @�@     �u@     �t@     �v@        
�
predictions*�	   `��Կ   ��
@     ί@!  ��K�B�) ��*$6@2���7�ֿ�Ca�G�Կ�K?̿�@�"�ɿ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���5�i}1���d�r�f�ʜ�7
������1��a˲���[����Zr[v��I��P=��pz�w�7��I��P=�>��Zr[v�>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�Š)U	@u�rʭ�@�������:�              �?              �?              �?       @       @      @      "@      0@      5@      B@     �F@      J@     �P@     �S@     �Z@     �[@     �`@      a@      c@     @c@     @a@     �a@      b@      b@     @`@     �_@     �Z@      [@     @Y@     @R@     �V@     �S@     �I@      O@      J@      H@     �C@      B@     �A@     �A@      7@      0@      4@      8@      0@      0@      *@      0@      (@      @      @      $@      &@      &@      @      (@      @      @      @      @      @       @      @      @      �?      @      �?      �?      �?      @               @               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?      �?      �?      �?      �?      �?              �?       @              @       @      �?      @       @               @      �?      @      @      @      @      @      &@      "@      @      @       @      @       @       @       @      &@      .@      (@      ,@      .@      1@      0@      (@      5@      7@      0@      6@      7@      4@      4@      9@      @@      7@      7@      0@      2@     �@@      &@      <@      (@      .@      .@      *@      ;@      >@      7@      "@      6@      7@      2@      "@      4@      $@      "@      *@      @      *@      @      (@      @      @              @      @      @       @       @       @      @              �?              �?       @              �?      �?              �?              �?        �0�>�2      ��ٞ	�J����A&*�d

mean squared error=�$=

	r-squaredL�$>
�L
states*�L	    ^��   @�@    ��NA!w�9Җ1�@)�쁛�DA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              G@     �w@      w@     �{@     ��@     ��@     p�@     ��@     �@     V�@     "�@     2�@    ���@    �l�@     ��@    ���@     ��@    ���@     ;�@     ?�@     ��@    ���@     ��@     �@    ���@    ��@    ��@    ���@    �I�@    ���@    @��@     ��@     ��@    ���@    �'�@    ���@     ��@     R�@    @	�@    ���@    ���@    `��@    �N�@     ��@    ���@    `�@    �C�@    ���@     {�@    @@�@     ��@     0�@    ���@     ��@    @i�@    `d�@     �@    ��@    @A�@     &�@    ���@    �r�@    @��@    @.�@     ��@     ��@    @%�@    �h�@    @%�@    ���@     c�@    �+�@     ��@     !�@     ��@     ?�@    �>�@    �/�@    ���@     4�@     ��@     �@     ��@     c�@     �@     в@     C�@     ��@     j�@     :�@     ��@     ��@     F�@     x�@     0�@     @�@     l�@     6�@     p�@     ��@     (�@     ��@     H�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     x�@     Ȋ@     Љ@     ��@     ��@     �@     8�@     ��@     p�@     ��@      �@     ��@     8�@     �@     �@     �}@     p{@     �{@     �y@     �w@      x@     �v@     �u@     �x@     @u@     �t@     u@      s@     �q@     r@      p@     pp@     �q@     �p@      o@     �q@     @n@     �i@     �h@      k@     �j@     @l@      i@      g@     �f@     �f@     �h@      f@     �c@     �d@     �c@     �c@      c@     @_@     �_@     �`@     @`@     @\@     �\@      `@     �_@     �[@     �`@     �\@     �[@      Z@      W@     �Z@     �Z@     @U@     @U@     @Y@     �S@     @U@      S@     �S@     @T@     @V@     @R@     �P@     @P@      R@     �O@     @P@     �P@      N@     @P@     �J@      L@      H@      M@     �L@     �G@     �F@     �F@     �@@      M@     �E@      H@     �F@      A@     �A@      K@     �A@      :@     �C@      F@      ?@      :@      ?@      ?@      A@      ?@     �C@      ?@      A@      4@      7@      ;@      :@      3@      ,@      ;@      4@      =@      8@      4@      8@      ;@      8@      7@      <@      ,@      .@      4@      1@      2@      2@      *@      *@      &@      2@      4@      (@      &@      (@      0@      ,@      &@       @      (@      ,@      *@      ,@      (@      (@      $@      (@      $@      @      $@      @      &@      (@      @      @      @       @      @      @      "@      @       @      $@      (@      @      "@      @      @      "@      "@      @      @      @      @       @      @      @      @      @      @     0s@     0s@      @      @       @      @      @      @      @      $@      "@      @      @       @      @      @      @       @      @      @      ,@      $@      $@      &@      @      &@      "@      ,@      "@      @      @       @      @      $@      (@      1@      @      @      (@      "@      *@      *@      1@      ,@      &@      @      9@      (@      6@      (@      &@      5@      *@      7@      7@      1@      0@      5@      7@      7@      8@      ?@      8@      ,@      3@      >@      3@      9@      4@      0@      6@      >@      7@      @@      ?@      =@      8@     �A@      F@      A@     �@@      B@      A@      >@      B@     �A@      ?@      ;@      6@      H@     �E@     �C@      E@      C@      L@     �D@      J@     �P@      J@      G@      O@      F@      K@      L@     �Q@     �M@     �I@      N@      P@     �R@     @R@     �N@      R@      Q@     �V@      W@      N@     @R@      X@      V@     �U@     �T@      Y@     @W@      Y@     @\@      \@     �Z@     @W@      `@      U@      _@     �]@      a@     @^@      `@     �`@     �a@     `a@     �`@     @b@      b@     @d@     @c@      d@     �e@     `g@     `h@     �e@      g@     �h@     �g@     �h@     @k@      i@     �j@      n@     �o@     `m@      n@     p@     `p@     Pq@     �q@     �r@     �r@      s@     0u@     @u@     0u@     v@      w@      x@      z@     �y@     �|@     0{@     P}@     0|@     �|@     @      �@     @�@     x�@     h�@     8�@     ؄@     ��@     �@     ��@     ��@     ��@     x�@     �@     ��@     ��@     ��@     x�@      �@     ��@     ��@      �@     ؘ@     ��@     ��@     �@     ,�@     ��@     ��@     У@     ��@     j�@     ��@     ��@     ��@     ��@     ��@     ձ@     %�@     ��@     g�@     �@     ��@     x�@     �@     ��@    �1�@    ���@     ��@     �@     ��@     ��@    �h�@    �{�@    @��@    �W�@    ���@    ��@    @��@    @��@    �o�@    ���@    `L�@    `m�@    �L�@     �@    ���@    �<�@    �x�@    ���@    `(�@    �H�@    `��@    ���@    ���@     ��@    @�@    `��@     g�@    �s�@    `��@    ���@    �I�@    �m�@    ��@    �,�@    �p�@     ��@    ��@    �p�@    @��@    ���@    `?�@    ���@    ��@    �x�@     ��@     H�@     #�@    @��@     1�@    ���@    �.�@    @E�@    �'�@    ���@    ���@     J�@    @��@     ��@    ��@    @_�@    ���@     h�@     q�@     �@     R�@     ��@      �@     ��@     ��@     @y@     `w@     `t@        
�
predictions*�	   ����   `�J	@     ί@!  �fmS@)@v+�1�?@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1�>h�'��f�ʜ�7
���(��澢f�����ߊ4F��>})�l a�>6�]��?����?f�ʜ�7
?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?ܔ�.�u�?��tM@h�5�@�Š)U	@�������:�              �?      �?      �?      @       @      $@      &@      5@      .@      3@      5@     �A@      @@      C@     �C@      E@      9@      =@     �A@      B@     �@@     �I@      @@     �D@     �C@     �E@     �C@      9@     �B@      7@      ;@      7@      ;@      8@      <@      .@      1@      @      $@      (@      "@      .@      (@      &@       @      @       @      @      @      @      $@      @              @      @      @      @      @       @      �?      �?      �?               @       @              �?               @               @              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?               @       @      �?       @      @      �?              @      @              @      �?      @       @      @       @      @      @      @      "@      @      @      "@      @      2@      &@      0@      4@      0@      4@      6@      6@      9@      5@      ?@     �C@     �D@      F@      F@      G@      N@     @Q@      Q@     @U@      Y@     @W@     @W@     @^@     �^@     �a@     �a@     �a@     �b@     �d@     �`@     @W@     �Z@      T@      P@      J@     �G@      B@     �A@      B@      >@      7@      :@      ,@      (@      ,@      $@       @      "@       @      *@      @       @      @      @      @              �?               @      �?               @      �?              �?      �?      �?               @              �?        �>z�2      u�	QL�����A'*�e

mean squared error<$=

	r-squared�&>
�L
states*�L	    ���   ���@    ��NA!&����@)y͒nbA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              E@     @x@     @|@     ��@     �@     �@     ��@     ��@     d�@     ��@     ��@    ���@    ���@     ��@    @}�@    @]�@    ���@     ��@    � �@    �k�@    �V�@     ��@    �F�@     ��@    ���@     '�@    ��@    ���@    ���@    �<�@    ��@     ��@     ,�@    ���@    @�@     ��@    @M�@    �B�@    �c�@    �j�@    �;�@     ��@    ���@    @��@    �I�@    ��@    ���@    @1�@    ���@    ��@    �U�@    ���@     k�@    �.�@    ���@    ���@    @=�@    ���@     �@    �5�@    �C�@     ��@    �&�@    @9�@    ��@    ���@    �N�@     ��@    @�@     G�@     ��@     ��@    �!�@     ��@    �
�@     8�@     4�@     ��@     ڿ@     t�@     �@     ޸@     f�@     ��@     ��@     ȱ@     Ͱ@     ��@     ��@     r�@     
�@     :�@     �@     ��@     ��@     �@     ��@     ��@     �@     ԛ@     H�@     ��@     $�@      �@     |�@     t�@     ��@      �@      �@     ��@     ��@     ��@     ��@     h�@     ��@     ��@     ��@     ��@     ؄@     ��@     ��@     �@     ��@     ��@     ��@     �}@     |@     `|@     py@     @{@     pz@     px@     pv@     Pv@     @u@     `t@     �t@     `t@     �s@     s@     �q@     q@     `s@     pp@     �q@     0p@     p@     �p@     �l@      n@     `l@     �k@     �h@     �f@     �k@     �d@      h@     �i@     �g@     �c@      f@      d@     �c@     @b@     �`@      c@     �c@     �`@     �`@     �_@      \@     �`@     �^@     �_@     @^@     �X@      X@     �Z@     @Y@     @]@      \@     @\@     �R@     �V@     �W@      Y@     �Q@     @V@     �S@     @R@      R@     @P@     �Q@     �N@     �P@     �Q@      N@     �P@      K@      O@     �J@      N@     �M@      J@      H@      C@     �H@     �I@     �F@     �H@      F@      I@     �E@     �C@     �F@      C@     �C@      E@      A@      A@     �@@      C@      =@      A@      @@      @@      >@      @@      7@      =@      4@      <@      9@      <@      :@      ;@     �B@      8@      7@      ?@      ?@      5@      :@      6@      7@      7@      (@      0@      :@      4@      .@      *@      1@      0@      &@      $@      5@      *@      "@      ,@      *@      1@      "@      $@      "@      (@      1@      0@       @      0@      "@      "@       @      3@      &@      ,@       @      "@      $@      *@      (@      ,@      @      @      "@      "@      �?      @      @       @      @      @      @      @      @      @       @      @      @      @      @      @     @w@     �w@      @      @       @      @      @      @       @      @      @      @      (@      (@      *@      @      "@      @      @      @      &@       @      @      @      &@      "@      (@      @      ,@      (@      "@      1@      &@      3@      1@      *@      .@      ,@      .@      *@      2@      .@      1@      *@      &@      7@      7@      (@      .@      *@      .@      1@      .@      8@      3@      ,@      4@      1@      4@      <@     �@@      8@      1@      :@      9@      8@      2@      :@      ?@      5@      9@     �B@      5@      1@      ;@     �B@     �@@      :@     �B@      <@      E@      G@      <@      D@      D@      D@      A@      F@     �E@      E@     �D@     �J@     �F@     �K@     �G@     �A@     �G@     �N@     �E@      J@     �H@      N@     �N@     @P@     @R@     �P@     �K@     @Q@      R@     �M@     �Q@     �K@      R@     �Q@     �U@     @P@     �R@     �W@     �V@      V@     �R@     �[@     @V@     �Y@     �Z@     �[@     @_@     �\@      [@     �[@     �`@     �_@     �]@     @b@     @]@     @a@     �_@      b@     `g@      d@      d@     �a@     @d@      e@      e@     �f@     �g@      e@     `f@      i@     �h@     @h@     �e@     `m@     `k@      n@      m@     `q@      p@     �p@     `q@     �q@     �q@     �r@     �r@      t@     �s@     �u@     @t@     @v@     �u@      v@     {@     �x@     p|@     �x@     �y@     �z@     �@     P�@     �@     �@     �@     �@     ��@     ��@     �@     �@     `�@     ��@     0�@     �@     8�@     @�@     p�@     �@     ��@     �@      �@     �@     ��@     �@     �@     ܚ@     ��@     ��@     8�@     ��@     ڡ@     �@     ��@      �@     ��@     :�@     �@     h�@     &�@     ��@     ñ@     γ@     y�@     F�@     j�@     &�@     :�@     ��@     6�@     ��@     ��@    �2�@    ��@    ���@     ��@    ���@     ��@    �J�@    �	�@     ��@    @}�@    @��@    @h�@    �i�@    �w�@    ���@    @��@    `��@    `��@    ���@    ���@    �$�@    `��@    ��@    �i�@     b�@    `��@    ���@    ���@    ���@    �I�@    ���@    ��@    `[�@    �8�@    @@�@    �I�@    @��@     {�@    �u�@     ��@    ���@    @�@    `'�@    ��@     ��@    �K�@     B�@    @��@    �1�@     ��@    @��@    �u�@    ���@    �P�@    @`�@    @P�@    ���@    �w�@    @��@    ���@    @A�@    �+�@    ���@     v�@     Ϸ@     �@     ��@     ��@     ޠ@     X�@     ��@     ��@     Ps@     �r@     pv@        
�
predictions*�	    ��ǿ   @+�@     ί@!  ���,/@)�R��|�8@2��@�"�ɿ�QK|:ǿ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�>h�'��f�ʜ�7
��ߊ4F��h���`�8K�ߝ�豪}0ڰ>��n����>>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?S�Fi��?ܔ�.�u�?�6v��@h�5�@�������:�              �?               @      @      @      &@      5@      <@      A@      E@     �H@     �L@     �Q@      N@     �P@     �S@     @T@     �J@     �U@     �P@     �S@     �S@     @P@     @R@     �Q@     �N@      M@     �L@     �D@     @R@      P@     �D@      F@      K@      C@      C@      8@      9@     �@@      5@      :@      6@      5@      .@      ,@      9@      $@      "@      0@      $@      &@      "@      @      "@      &@      @       @       @      @      @      @      @       @      �?      @              �?              �?      �?      @       @      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      @              �?      @       @      �?      @      @      @      @      @      @      @      @      @       @       @      @      @      "@      @      "@      .@      0@      4@      $@      5@      &@      *@      ;@      4@      5@     �C@      :@     �@@      ?@     �@@     �D@     �@@     �F@      C@     @P@     �C@     �J@      S@     �O@     �Q@     �R@     �T@      O@     �M@      O@     �M@      L@      L@     �J@     �D@     �G@     �D@     �E@     �C@     �@@      :@      :@      6@      0@      0@      &@      @       @      $@      "@      @      @      @      �?      @       @               @       @      �?       @      �?              �?      �?               @              �?              �?              �?        fg���2      ���C	C�����A(*�e

mean squared error�1%=

	r-squared�|!>
�L
states*�L	   ����   �`�@    ��NA!X>{�u\�@)=TZVdA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              I@     �w@     Pv@     `{@     X�@     H�@     H�@     X�@     �@     ԫ@     A�@    ���@    ���@     �@    ���@    �I�@    @G�@    @��@    �,�@     �@    ��@    �}�@    ���@    ���@     (�@     {�@    ���@    @��@    �8�@    ��@    @��@    @��@    �}�@     ��@    ���@    �H�@     ��@    ��@    @�@    ��@    ��@     s�@    �m�@     (�@    ���@    `Y�@    ���@    ���@     T�@    �
�@     ��@    @J�@     ��@    @��@     d�@     ��@    �m�@    @��@     ��@    @��@    �;�@    @�@    �J�@    @��@    ��@     ��@    �?�@    ���@     ��@     <�@    �>�@    �5�@     /�@     ��@    �L�@    ���@     �@     ƿ@     x�@     m�@     <�@     -�@     6�@     ��@     c�@     �@     &�@     P�@     
�@     ƨ@     �@     ��@     t�@     H�@     \�@     ��@     �@     h�@     �@     ܘ@     ��@     (�@     ��@     |�@     X�@     ��@     H�@     t�@     x�@     P�@     ��@     8�@     P�@     H�@     ��@     h�@     ��@     ؂@     ��@     P�@      �@     ��@     �@     �~@     �@     @@     p@     �}@     0y@     P{@     0y@      v@     @x@     �v@      u@     pv@      v@      u@     �q@     �p@     �q@     �r@      p@     Pp@      p@      o@      q@     �m@      j@     �k@      k@     �k@     @h@     �g@     �g@     �g@     �d@     �g@     �f@     �c@     �f@     �d@     �`@     `c@     �`@     �d@     @e@     �a@     �_@     @^@     �[@     �[@     @a@     �Y@     ``@      \@      X@     �W@     �W@      W@      X@     �W@      X@     @W@     �Y@     �R@      S@     �T@      L@      J@     @U@     �O@     @P@     �Q@     @Q@     �O@     �O@     �Q@     �O@      Q@      L@     �L@     �P@     �H@      P@     �L@     �O@     @P@     �F@     �L@      C@      C@      G@      M@      F@      E@     �D@      @@      @@     �C@     �A@      <@      F@      ?@      C@      =@      A@      :@      @@      3@      ?@     �@@     �@@      ?@      ;@      4@      >@      9@      ;@      7@      8@      1@      1@      ;@      4@      5@      ,@      (@      6@      5@      &@      4@      (@      "@      3@      &@      @      *@      3@      &@      .@      $@      .@      *@      (@      2@      .@      2@      .@      (@      &@      @      &@      $@      &@      $@      $@      @      @      @      &@      *@      (@      @      @      &@      @      "@      $@      @      @      @      @       @      "@      $@      @       @      "@       @      @       @     �u@     �w@       @      "@      @      @      $@      $@      @      @      @       @       @      $@       @      @      @      &@      $@      &@      ,@       @      @      @      "@      @      @      @      (@      0@      ,@      (@      &@      &@      (@      *@      ,@      "@      $@      ,@      4@      2@      7@      0@      7@      2@      3@       @      1@      *@      0@      =@      1@      3@      .@      7@      4@      8@      0@      5@      4@      4@      ;@      ;@      :@      ;@      9@      A@      <@      9@      C@      E@      9@      <@      @@      =@      D@     �C@      F@      C@      F@      C@      B@      F@     �B@     �A@     �B@     �F@     �M@      J@     �A@     �J@     �F@     �B@      G@      L@      P@     �O@     �L@     @R@      L@      L@     �K@      M@     �S@     @R@     �Q@      R@     �Q@     @S@     @R@     @S@      R@     �U@     �R@     �S@     �T@      Y@     �X@     �Y@     �X@     �X@     @[@     �Y@     @^@     �X@     �]@     �Z@      `@      a@     �`@     @_@      b@     @a@     @]@      b@     �b@      b@     �c@     �d@     �d@     �b@      e@     `e@     �b@     �d@     �h@     �e@     �e@     �j@      j@      i@     �j@     `h@     �n@     @m@     �l@      q@     @q@     �q@     �p@     �q@     �q@     �r@      r@     0s@     �t@     �t@     �t@      v@     `v@     0x@     px@     �z@     �w@     �{@     P{@     `~@     �{@     P@     �@     8�@     ��@     ��@     P�@     �@     8�@     (�@     (�@     ��@     ��@     �@      �@     x�@     �@     ��@     ��@     �@     L�@     T�@     <�@     Ж@     ��@     ��@     X�@     8�@     ��@     �@     Π@     ��@     d�@     x�@     ��@     T�@     z�@      �@     ^�@     ʮ@     o�@     �@     ��@     ��@     y�@     �@     ��@     p�@    ��@     ��@     *�@    �y�@    ���@    ���@    �^�@     ��@    �J�@    ��@    ���@    �g�@    @d�@    ���@    ��@    �;�@    �h�@    `��@    ���@    `��@     ��@    �t�@    �n�@    `��@    ��@    `��@    ���@     ��@    �:�@     v�@    `|�@     r�@    �;�@    ���@    �<�@     [�@     \�@    `%�@    ���@    �W�@    ���@    ���@     ��@    ���@    �-�@    `�@    �,�@    `�@     p�@    ���@     ��@    �v�@    ���@    @9�@    ���@    �"�@    @��@    ���@     ��@    ��@    @�@    ���@    �A�@    �>�@     �@    ���@    ���@     ��@     ��@     j�@     Τ@     ��@     ��@     ȍ@     X�@     �q@     �m@      y@        
�
predictions*�	   �Dfǿ    �K@     ί@!  ,���=�)����PB@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x���ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?{2�.��@!��v�@�������:�              �?      �?      �?      �?      @      &@      @@      <@     �D@     �I@      M@      R@     @T@     @[@     �Z@     @\@     �X@     �]@     @_@     �Z@      Z@      Z@      Y@      ^@     �X@     @Z@     @\@      T@      S@     �O@      O@      G@      J@      L@      K@      ?@      @@      ?@      @@      A@      5@      0@      *@      (@      6@      (@      $@      ,@      .@      &@       @      @      *@      @      @       @       @      @      @      @      @      �?      �?      @      @      �?       @      �?       @      �?      �?      �?      @              �?       @              �?              �?              �?              �?              �?              �?      �?               @               @              �?       @       @       @      �?               @      �?      �?      �?       @      �?      �?      �?      �?       @      @      @      @      @      @      $@      @      @      &@      $@      @       @      &@      (@      &@      2@      1@      1@      ,@      2@      6@      2@      6@      4@      :@      6@      ?@     �@@      :@     �A@      D@     �D@     �A@      ;@      D@      G@     �E@      A@      8@      A@      A@     �B@     �E@     �A@      ;@     �@@      6@      :@      0@      3@      9@      (@      *@      3@      "@      &@      (@      @      $@      @      @      @       @       @              @      �?      �?       @              �?              @              �?      �?              �?              �?              �?        �h=t�2      �m�	�������A)*�e

mean squared error�;=

	r-squared؝@>
�L
states*�L	   �[��    g�@    ��NA!qé,���@)l�+��sA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              U@     �v@     �u@     �z@     �@     <�@     P�@     ��@     ��@     >�@     *�@    ���@    ��@     �@     /�@    ���@    ���@    ��@     ?�@     f�@    �e�@    �:�@    �V�@    ���@    ���@     ��@    @3�@    ���@    ��@     ��@    ���@    ���@    @��@    �;�@    @H�@    �r�@    ���@    ��@    ���@    ���@     ��@     ��@    @��@    ���@     ~�@    `��@     �@    �U�@    @X�@    `��@    ���@     ]�@    ���@    ���@     ��@     5�@    `��@     <�@     ��@     ��@     ��@    @��@    ���@    �1�@    @y�@    ���@    ��@     ��@    �Z�@     d�@    ��@    ���@    ���@    �J�@     ��@     ��@     Z�@     ο@     V�@     �@     y�@     ��@     ��@     �@     6�@     ]�@     ��@     ��@     �@     ��@     (�@     *�@     ܤ@     �@     *�@     ��@     ��@     ��@     ��@     �@     �@      �@     �@     �@     @�@     <�@     ��@     ��@     �@     H�@     �@     p�@     ��@      �@     ��@     ��@     ��@     ��@     x�@     h�@     ��@     Ȃ@     P�@     �@     �}@     �~@     �}@     �y@     z@     �z@     p{@     �w@      w@     `w@     �u@      v@     s@     @v@     t@     �r@     �q@     �p@     0p@     s@     �p@     Pq@     @p@     �q@     `o@     �i@      k@     �k@     `k@     `i@      j@     �f@      i@     �g@     �j@      g@     @c@      d@     @`@      c@     �c@     @d@     �e@     �c@     �a@     �a@     �a@      Y@     �Z@      `@     �Y@      Z@     @_@     @^@     �Z@     �X@     @Y@     �Y@     @T@     �V@      W@     �U@     @U@     �T@      V@      W@     @R@     �T@     �P@     �Q@     @R@      R@     �P@      M@      L@     �N@     @Q@     �P@     �L@      Q@     �N@      F@      K@      M@     �N@     �F@     �J@     �I@     �I@      J@      G@     �F@     �D@     �D@     �A@      C@     �E@      E@     �G@      C@      =@     �D@      <@      G@      >@     �C@      B@      9@      6@      :@      @@      7@      ;@      ?@      5@      ;@      7@      3@      4@      5@      ?@      7@      7@      5@      5@      8@      0@      .@      4@      5@      (@      $@      2@      *@      .@      .@      &@      9@      $@      1@      @      $@      .@      0@      $@      *@      3@      (@      (@      (@      *@      .@      @      "@      &@      "@      "@       @      "@      "@       @      (@      ,@      "@      @       @       @      "@      ,@      @      @      @      &@      $@      @      @      @      @     �w@     px@      .@       @      $@      @      @      @      "@      "@      &@      &@      @      @      @      @      @      *@      0@      &@      4@      0@      &@      @      "@      *@      *@      $@      4@      (@      @      "@      2@      0@      $@      .@      (@      4@      *@      4@      2@      6@      3@      1@      6@      4@      7@      :@      8@      1@      3@      3@      :@      ;@      1@      1@      5@      1@      ;@      >@      8@      7@      =@      >@      ?@      8@      =@     �A@      ?@     �A@      A@      =@     �@@     �B@      ;@      C@     �C@     �D@      C@     �C@      C@     �D@     �I@     �H@     �E@      L@     �H@      J@     �L@      J@     �N@      Q@      M@      N@     �E@      O@      N@     �P@      K@      Q@      K@     �Q@     �M@     �U@     �Q@     �Q@      M@      L@     �V@     @U@      X@      S@      V@     �W@     �V@     �X@     �W@     @Z@     �Y@     �X@     �S@     �U@     �^@     �[@      [@     �_@      ^@     �`@     �`@     ``@     `c@     �d@     �a@      b@     @a@     �b@      e@     @c@      d@      f@     �e@      g@     �c@     @g@     �i@     @i@      l@      j@      i@     `k@     �i@     @m@     @l@     `n@     p@      m@     �r@     �q@     @n@     pr@     �r@     Pq@     s@     �t@     �t@     u@     �t@     �t@     �t@     �v@     �y@     �y@     �y@     `{@     �{@     �}@     �{@     H�@     ��@     x�@     H�@     ��@     ��@     x�@     ȁ@     @�@      �@     ��@     ��@     (�@     ȉ@     8�@     �@     x�@     8�@     �@     ��@     ��@     �@     Ԕ@     d�@     ��@     ��@     |�@     8�@     p�@     ��@     ��@     T�@     ��@     .�@     ^�@     �@     0�@     ��@     �@     �@     8�@     ��@     ��@     A�@     ݶ@     )�@     m�@     �@    �X�@     ��@    �\�@     �@     :�@     ��@    �R�@    �x�@    �9�@    @a�@    ���@    ��@    @��@    �t�@    �g�@    �G�@    �s�@    �9�@    ��@    �
�@     ��@    ���@    @!�@    `��@     ��@    ��@     �@    �m�@    ���@    ��@    `z�@    �g�@    `��@    �-�@    `��@    ���@     ��@     ��@    ���@     �@     ��@     [�@     ;�@    ���@    ��@    ��@    �+�@    @V�@    ��@    `y�@     �@     ��@     ��@     ��@    ���@     `�@    ���@     �@     ��@    ���@    �*�@    ��@    �.�@     ��@     ��@    ���@    ���@    ���@     ҷ@     u�@     z�@     ��@     ��@     |�@      �@     Ѓ@     �u@     �l@     �{@        
�
predictions*�	   �ȸƿ   �v�@     ί@!  㽐=P@)H��	?@2��QK|:ǿyD$�ſ%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������jqs&\�ѾK+�E��Ͼ��~��¾�[�=�k����[�?1��a˲?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?u�rʭ�@�DK��@�������:�              �?              �?      @      @      @      .@      4@      3@      ?@      :@     �D@      E@      D@      P@      H@      L@      F@      G@      L@      K@      H@     �G@      J@     �G@     �D@      J@      B@     �C@     �A@      =@      <@      >@      1@      7@      5@      3@      1@      0@      5@       @      *@      2@      @      @      $@      "@      &@      @      "@       @      @      @      @      @      @      @      @       @      @      �?       @       @      �?              �?       @              �?      �?       @       @      �?      �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              @              �?              �?       @      �?      �?       @      �?       @              @       @      @       @      @      @      @      @      @      @      @       @      $@      @      ,@      *@      &@      $@      1@      (@      0@      1@      4@      6@      ;@      6@      5@      >@      A@     �I@      B@     �H@     �H@     �E@     �N@     �L@     �R@     �V@     �X@     @W@     @^@     �X@     �a@     �^@     @\@      \@     @Z@     �Z@      S@     �Q@     �S@      N@     �K@      F@     �A@     �A@      8@      ;@      :@      5@      4@      7@      &@      "@       @      @       @      @      @      @      �?      �?      @       @      @      �?      @       @       @       @              �?               @      �?              �?              �?        \���24      e��	PP����A**�h

mean squared error:%=

	r-squared�j">
�L
states*�L	   @���   @5�@    ��NA!n�Ǭ�&�@)0�h���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              `@     �z@     �t@     P|@     ��@     ,�@     ��@     ܟ@     ��@     (�@     y�@    ���@    ���@     ��@    �&�@    @��@    ���@     ��@     k�@     i�@     ��@     ��@    ���@    �5�@    ��@    ��@     ��@    @��@     ��@     y�@    ��@    ���@    @b�@    ��@    @��@    �4�@    ���@    �q�@     ��@    @o�@    @��@    ���@    ���@    �q�@    @=�@    ��@    ���@     2�@     ��@    ���@    �_�@    `��@    @��@    @ �@    ��@    �u�@    `��@    ��@    ���@     K�@     ��@    ��@    �h�@    ���@    �
�@    �8�@    @��@    �N�@    ���@    �[�@    �w�@    ��@    ���@     ��@    ���@    �B�@    �x�@     r�@     U�@     H�@     �@     &�@     �@     ��@     �@     ��@     ذ@     b�@      �@     ��@     ̨@     �@     ��@     t�@     (�@     n�@     *�@     ��@     $�@     ؛@     �@     �@     ,�@     @�@     t�@      �@     Ԑ@     �@     p�@     ȏ@     ��@     �@     (�@     ��@     @�@     0�@     `�@     ��@     p�@     0�@     ��@     ��@     P�@     ��@     `�@     �{@     �{@     �{@      z@     @z@     �x@     �w@     �w@     �w@     `v@      v@     �t@     `r@     �s@      r@     Ps@     �p@     �q@     pq@      s@     @n@     0p@     `o@      k@     `n@     @m@      k@     �i@      j@      f@     `h@     `i@      d@     @e@     �d@      c@     �c@     �b@     `d@      d@      b@      a@      ^@     �`@      a@     @\@     �^@     @\@      ]@     @Z@     @_@     `a@     �Z@      Z@      Y@     @Z@     �V@     �Y@     �X@     �P@     �V@     �X@     �R@     @T@     @U@     @S@     �S@      M@     �Q@      N@     �R@     @S@     �Q@     �Q@     �N@      N@      H@     �L@      L@     �N@     �M@      K@     @Q@     �M@      G@      G@      C@      C@      M@     �G@      E@      D@      B@      C@      H@     �E@     �C@      A@      ?@      @@      ?@     �F@      ?@      @@      >@      B@     �B@      <@      B@      ?@      A@      8@      :@      ?@      :@      ,@      ;@      5@      5@      9@      <@      5@      :@      6@      6@      <@      5@      3@      9@      ,@      &@      3@      0@      5@      1@      "@      $@       @      (@      $@      *@      &@      $@      (@      0@      5@      &@      0@      $@      ,@      @       @      ,@      ,@      "@      $@      $@      @      .@      @      @      (@      &@      @      @       @      @      @      @      @      "@      @      @      @      @      @       @     �w@     `{@       @      "@      (@      @      &@      @       @      &@      @      .@      &@      $@      (@      @      &@      2@      ,@      (@      0@      ,@      (@      $@      0@      *@      1@      $@      $@      *@      1@      2@      0@      0@      5@      5@      1@      1@      .@      0@      .@      2@      1@      2@      8@      5@      7@      7@      9@      8@      6@      9@      =@      4@      <@      >@      ;@      :@      ?@      6@      =@      ;@      @@      @@      9@      >@      E@      >@      G@      F@     �J@      D@     �E@     �B@      7@      B@     �E@      J@      G@      F@     �K@      D@      G@      J@     �G@     �G@     �F@      L@      P@     �M@     �L@     �O@     �P@     @P@     �N@      N@     @P@      P@     �R@      S@     �R@     �S@     �P@     �U@      O@      Q@     @P@     �P@     �Q@      W@     �V@     �V@     �X@     �S@     �U@     @U@     �X@      Y@     �[@     �Y@     �\@      [@     �[@     @`@     �`@     �]@     @]@      [@     @_@     �`@     @e@     @`@     `d@      d@     `d@     �d@     �f@     @a@     �c@      c@     �e@     @f@     �h@     `j@     �g@      m@     @h@     �k@     `l@     `l@     @o@      p@     �n@      p@      o@     �q@     @q@      p@     �p@     0q@     0r@     �r@     �q@     v@     `x@     �v@     `u@     0x@     `y@     �w@     �y@     �{@     p{@     �|@      ~@     �~@     �@     �@     �@     @�@     �@     �@     ؂@     ؃@     `�@     �@     ��@      �@      �@     ��@     Ћ@     �@     h�@     ,�@     ̐@     $�@     ��@     p�@     Д@     (�@     h�@     p�@     X�@     0�@     \�@     ��@     b�@     ܠ@     B�@     R�@     ��@     �@     ��@     ̪@     L�@     D�@      �@     s�@     ��@     س@     �@     ķ@     9�@     ��@     \�@    �o�@     ��@    ���@    ���@    �x�@    ���@    �2�@     8�@    ���@    �>�@    ���@    @_�@     F�@    @��@    @��@    @��@    ���@    � �@     
�@    ���@    `��@    ���@    �H�@    �}�@    `��@     ��@    ���@    `�@    @��@     ��@    `q�@    ���@    ���@     Q�@    ���@     ��@    ��@    ���@    ���@    ��@    @y�@    �H�@     R�@    �s�@     ��@    ���@    �@�@    @>�@    `R�@     ��@     ��@    �v�@    ��@     �@    ���@    @��@    �V�@    @�@    ��@    �1�@    ���@    � �@    @G�@     A�@    @��@    �.�@    ��@     �@     ��@     ��@     ~�@     N�@     v�@     �@     ��@     x�@      z@     pq@      ~@        
�
predictions*�	    ��Ϳ   �$@     ί@!  8��7@)�lE�7@2��Z�_��ο�K?̿�@�"�ɿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=����(��澢f�����_�T�l׾��>M|Kվw`f���n>ہkVl�p>�f����>��(���>a�Ϭ(�>8K�ߝ�>1��a˲?6�]��?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�6v��@h�5�@�������:�              �?      �?              �?              �?       @      @      @      .@      1@      C@      E@      G@      E@     �K@      P@      V@      T@     �S@     @R@     @R@      Q@     �T@      R@      P@      S@      S@     @R@      M@      I@     �H@      H@      O@     �D@     �F@      >@      >@      >@     �G@      :@      :@      @@      <@      *@      <@      $@      *@      ,@      "@      $@      $@       @      @      @      "@       @      @       @      �?      @      �?      @       @      @       @       @      @              @      @               @               @      �?       @      �?               @              �?       @              �?       @              �?              �?      �?              �?              �?              �?              �?              �?               @              �?              �?               @      �?      �?       @       @       @              �?      @      �?       @              @      �?       @      @      @      @      @      @       @      @      @      @      @      "@      @      @      @      &@      0@      1@      .@      0@      2@      (@      3@      1@      3@      ?@      ?@      @@      :@     �E@     �H@     �G@     �I@      M@      I@      J@     �R@      R@     �S@     �R@      R@      Q@     �P@     �U@     �S@     �M@      P@     �P@      J@      E@     �@@      >@     �@@      A@      8@      8@      7@      5@      .@      6@      "@      *@      $@      @       @       @      @      "@      @      �?      �?      @               @       @      �?      �?       @              �?      �?      �?              �?              �?              �?      �?              �?        �,��B2      ��}�	�R����A+*�d

mean squared error)!=

	r-squared�6>
�L
states*�L	   ���   @Q�@    ��NA!H~�&R�@)�y���	A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             @Y@     �~@     pt@     �x@      �@     4�@     |�@     J�@     Σ@     ��@     S�@     ��@    ���@     ��@    �S�@    ��@    @t�@     ��@     ��@    ���@     X�@     M�@    ��@     ��@    �q�@    �0�@    ���@    @Q�@     ��@    ���@    �)�@    �3�@    ��@    ���@    ��@     ��@    ���@    �9�@    @��@    ��@    @q�@    �a�@     "�@     ��@    @_�@    ���@     ��@    @��@    ���@    `��@    `��@    �`�@    �O�@    @�@    ���@    �%�@    ���@    ���@    @��@     U�@    @�@    �@�@     ��@    @�@    ���@    @�@     ��@     �@      �@     V�@    ���@     h�@     ��@    ���@    ���@     L�@    ��@    �6�@     ɾ@     -�@     ��@     Ƿ@     Ŷ@     d�@     Ӳ@     ��@     ��@     t�@     z�@     (�@     ֨@     ��@     *�@     `�@     z�@     ̡@     �@     �@     ��@     H�@     0�@     `�@     ؖ@     `�@      �@     ��@     �@     0�@     ��@     ��@     P�@     @�@     0�@     �@     ��@      �@     ؆@     �@     8�@     h�@     ؂@     ��@     p�@     ��@     0�@     p~@     `}@     |@     �{@     �z@     pw@     �w@     �w@      w@     �t@     0u@     @u@     �u@      s@      r@     ps@     �q@     pr@     �r@     0q@     �p@     Pr@      m@      o@     �j@     �m@     `l@      l@     �j@     @j@     �g@     �d@      i@     �h@      e@      g@      c@     `b@     �a@     `c@     �c@     �_@     ``@      `@     �`@      [@      ^@     �]@      `@      a@     �X@      `@     �S@     �W@      W@      W@      Y@     �P@     �V@     �V@     �W@     @Z@     �R@     �S@     �S@     @U@     �S@     @Q@     @Q@     �P@     @Q@     @Q@     @Q@     �P@      I@      P@     �H@     �L@     �E@     �O@     �K@      N@     �J@      F@     @R@     �F@     �E@     �G@      C@     �B@     �A@     �K@     �B@     �@@      =@      <@     �B@     �F@     �@@      D@      =@      @@     �@@      @@      @@      =@      ;@      3@      ;@      >@      9@      6@      C@      :@      =@      =@      9@      4@      ,@      4@      6@      ,@      0@      5@      0@      *@      *@      7@      0@      4@      3@      2@      .@      &@      (@      (@      .@      &@      2@      @      (@      *@      ,@      2@      (@      *@      "@       @      .@      "@       @      (@      (@      $@      "@      (@      @      0@      @      *@      @      @      @      @      @       @       @      @       @      @      @      @      $@      @              @      "@     `x@     �~@      @       @      &@      "@      $@      .@      @      @      "@      (@      $@      "@      @      *@      "@      (@      .@      (@      &@      &@      $@      @      .@      @      .@      (@      &@      (@      ,@      ,@      $@      (@      2@      1@      *@      (@      3@      ,@      3@      5@      0@      8@      4@      3@      2@      ,@      1@      ;@      :@      :@      3@      :@      >@      8@      9@      =@      9@      >@     �D@      @@      7@      8@      ?@      A@      @@      A@     �F@     �C@      >@     �D@      ?@     �H@      E@      D@      A@      :@      G@     �C@      F@     �F@     �A@      L@      J@      F@      H@      G@     �N@     �R@      L@      P@      K@      N@     �R@      P@     @S@      L@      O@     �O@     �P@     �T@     �M@     �K@     �R@      S@     @U@      R@     �T@     �V@     �Y@      U@     �W@     �Z@     @^@     �T@     �[@     �V@      _@     @Y@     @]@      _@     �X@      _@     @a@     `a@     �`@     �b@     �a@     �b@     @a@     �c@      c@     �d@      d@     �d@     �d@     �d@      e@     @g@     `k@      i@     �h@     @j@     `k@      k@     �j@      k@     �k@     �k@     �p@     `n@     �l@     r@     0q@     pq@     `r@     �r@     `t@     �s@     �t@     �t@      u@     �t@     @w@     �u@     `v@     �x@     @y@     �x@     �z@     �{@     Py@      |@     ��@     �~@     0�@     �@     ��@     ��@     8�@     @�@      �@     H�@     �@     ��@      �@     ȇ@     @�@     �@     ��@     ��@     ��@      �@     �@     ��@     Ԓ@     ��@     ��@     x�@     @�@     ,�@     Ț@      �@     @�@     �@     �@     �@     8�@     
�@     ƥ@     ��@     (�@     ��@     ��@     b�@     `�@     �@     y�@     B�@     `�@     �@     ��@     	�@     ^�@     L�@     ��@    �c�@    ���@     ��@     2�@    ���@     �@     ��@    ��@     Y�@    ���@    @T�@    @!�@    ���@    �,�@    ��@     5�@     k�@    �T�@     ��@     ��@    �%�@    `��@     �@    �w�@     �@    ���@    `�@    `F�@    ���@    ��@    �%�@    �#�@    ���@     ��@     ��@    ���@    @��@     {�@     ��@     p�@     S�@     C�@     ��@    @��@    `0�@    `3�@    ���@     p�@    @�@    �%�@    @��@    ���@     ��@    ���@    @��@    ���@     {�@    ���@    ���@    @z�@    �R�@    �o�@    �	�@    @��@    ���@     ��@     4�@     �@     r�@     ~�@     h�@     \�@     ��@     ��@     �|@     �n@     }@        
�
predictions*�	   �<�Ŀ   @W|@     ί@!  ��>�)u��UID9@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1�f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?3?��|�?�E̟���?yL�����?w`<f@�6v��@�������:�              �?      �?      @      @      .@      9@     �@@     �K@      L@     �Q@     @Y@      Z@     �]@     �[@     �\@      a@      a@      `@     @c@      ^@     �\@     �^@     �]@     @V@     �U@     �Q@     @V@      Q@      K@      N@      H@      D@      E@     �@@      1@      6@      :@      3@      ?@      2@      3@      *@      (@      (@      @      "@      $@      $@      @      "@      @      @      @      @      @      @              @       @       @       @       @      @      �?      �?               @      �?              �?      �?       @       @      �?              �?               @              �?              �?               @       @               @      @      @               @      �?      @      @      �?       @      @      @      �?      @      @      @       @      @      @      "@      @      @       @      &@      "@      "@      $@      (@      (@       @      *@      &@      0@      1@      6@      .@      6@      ;@      :@     �@@      8@      5@     �B@      :@      =@      E@      >@     �D@      D@      @@      C@      F@      >@      <@      <@      <@     �C@     �@@     �A@      C@      4@      7@      .@      1@      4@      ,@      *@      2@      @      @      (@      @      @      @      @              @      @              @      �?               @       @      �?      @      @              �?              �?              �?      �?              �?        bPJ��2      ���C	�{�����A,*�e

mean squared errorF� =

	r-squared��7>
�L
states*�L	    ���   ��@    ��NA!�AqA5�@)��(g�DA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �T@     �}@     @y@     �|@     ��@     x�@     �@     �@     ��@     �@     '�@    ���@     ��@    @��@    ��@    ���@    ��@    @��@     ��@    ���@     ��@     ��@     ��@    ���@    ���@     ��@    ��@    �h�@     ^�@    @��@    ���@    @��@    ���@    @��@     i�@    ���@     Y�@     ��@     �@    ��@     ��@    �v�@    @�@     ��@    �D�@    ���@    ��@    ��@    `��@    �i�@     ,�@    ��@    ���@    `��@     g�@    ��@    ���@     ��@    � �@     ��@     "�@     9�@    ���@    ���@    @j�@    @�@    ���@    �"�@    @��@     s�@     ��@     ��@    �z�@    ��@    ���@     *�@    ���@    ���@     ��@     ��@     й@      �@     ��@     ��@     ��@     f�@     ��@     n�@     R�@     *�@     ̨@     0�@     ��@     ��@     �@     �@     ܠ@     �@     8�@     ��@     ��@     ��@     ��@     ��@      �@     ؔ@     В@     L�@     ��@     �@     p�@     ��@     ��@     @�@     ؉@     @�@     Љ@     Ѕ@     H�@     x�@     ��@     ��@     ��@     @�@     �@     �@      }@      @     �~@     �y@     �{@     �v@     �t@     Pw@      v@     �t@      w@     �s@     �s@      t@     @s@     0q@     r@     `r@     �p@     �p@     pq@     �p@      l@     @n@     �k@      l@     `l@      g@     �k@     �k@     `h@      h@      f@     @c@     �e@     �d@     @e@     �c@     �`@     �a@     `a@     @b@     �a@     �b@     �a@     �^@      [@     �`@     @]@     @^@     �`@     �Z@     @W@      Z@     �W@     @W@     �Z@     �Y@     @U@     �Q@      U@     �T@     @W@     �S@      U@     �R@     @R@     �R@      Q@      O@      Q@     �M@      I@     �K@     �O@     �Q@      L@     �M@     �K@      J@     @R@      L@      L@      K@      H@     �F@     �G@     �N@     �D@     �E@     �E@      C@     �F@     �D@      <@      8@     �A@     �A@      8@      =@      C@      :@      =@      C@      7@      @@     �@@      =@      >@      <@      <@      <@      2@      <@      8@      9@      ;@      7@      :@      2@      9@      <@      (@      0@      3@      (@      5@      .@      0@      .@      *@      "@      3@      *@      4@      ,@      6@      (@      ,@      "@      .@      (@      $@      ,@      "@      &@      $@      $@       @      *@      $@      .@       @      &@       @      "@      $@      (@      ,@      @      "@      $@      @      &@       @       @      @       @      @      $@      *@      @      @      @      @      ,@     z@     �@      ,@      @      $@      @      1@      @      @      (@      $@      "@      6@       @       @      *@      (@      "@      $@      0@      ,@       @      (@      $@      1@      2@      &@      *@      ,@      &@      ,@      *@      *@      @      ;@      "@      9@      *@      8@      .@      1@      7@      ,@      3@      6@      5@      ;@      7@      6@      2@      ;@      @@      2@      7@      A@      B@      ;@     �A@      :@      <@      B@      <@      >@      @@      6@      E@      B@      C@     �@@     �A@      A@     �D@      C@      H@     �G@      B@      @@      F@     �A@     �E@      A@      H@      J@      C@     �I@      K@      K@      L@      I@      Q@     �M@     �P@     �M@     �J@     �P@     �P@     �P@      R@      U@     �R@     �U@     �U@      X@     �Q@     �S@     �T@     @W@     @T@     @W@     �X@     �U@     �U@     @Z@     �X@     �Y@      [@     @[@     @Z@      Y@     �]@     �^@     �\@     @\@      \@      b@     �\@     @`@     `b@     @b@     �b@     �d@     �b@     �c@     �a@     �e@      d@     `g@      f@      g@     `h@      f@     �h@     �g@     �j@      k@     �l@      k@     �m@     �l@     `n@     �m@     @k@     pp@     @p@     �r@     q@     Pq@     Pv@     �s@     pt@     �t@     �t@     �x@     �w@     �v@      w@     �y@     �y@     @y@     �|@     �@     �{@     �|@     �~@     `@     ��@     �@     P�@     ��@     @�@     ��@     `�@     �@     h�@     ؄@     8�@     H�@     �@     H�@     h�@     H�@     x�@     0�@     8�@     �@     h�@     �@     L�@     �@     0�@     ܖ@     l�@     ,�@     ț@     ��@     ��@     ~�@     �@     �@     �@     ֤@     :�@     ʨ@     ȩ@     ��@     ��@     ذ@     6�@     ��@     :�@     $�@     h�@     �@     ��@     ɽ@     ѿ@     ��@     �@     ��@    �&�@    ���@     ��@    � �@    �N�@     �@     ��@    ���@    �:�@    ���@    �l�@     ��@    @��@    @�@    �f�@    �y�@    @o�@    �x�@     �@    ���@    @��@     /�@     ?�@    �R�@    @u�@     +�@    @r�@    ���@    @�@    `��@    ���@     q�@    ���@    �O�@     W�@     D�@    ���@    �k�@    ��@    �8�@     ��@    ���@    `/�@    �)�@     ��@    �b�@     ��@    @��@    @5�@    ���@    ���@    ���@    �!�@    ���@     ��@    ���@    �C�@    �+�@    ��@    @��@    �K�@    ��@    ���@    �.�@     ��@     ��@     &�@     ��@     ̣@     �@     \�@     ��@     ~@      s@     �~@        
�
predictions*�	   `�j¿   �:k	@     ί@!  pc��!@)\!���zC@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1�>�?�s���O�ʗ������%�>�uE����>1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�Š)U	@u�rʭ�@�������:�              @              &@      1@      4@      =@      G@      F@     @V@     �T@     �X@     �^@     @Z@     �[@     �Y@      [@     �^@     �\@      _@     �Z@     @V@     �Y@     @U@     �U@     �P@     �R@      H@      J@     �D@      F@      <@     �@@     �B@      >@      ;@      6@      8@      6@      4@      $@      2@      ,@      $@      $@      (@      $@      @      &@      @      @      @      @      @      @      @      @      @      @      @      �?      �?      �?      �?       @       @      @               @      �?       @               @      �?              �?               @       @              �?              �?              �?              �?              �?               @               @      �?      �?      �?      �?       @       @       @      @      �?              �?      @       @      @       @              �?      @      @       @      @      @       @       @      @      @      @      (@      @      (@      @      (@      0@      .@      "@      .@      2@      3@      8@      5@      =@     �@@      1@      ?@      A@     �B@      >@      6@      ?@      @@      D@      B@     �B@     �I@      N@     �E@      G@     �D@     �M@      J@      G@     �E@      @@      @@     �B@      B@      @@      ?@      >@      7@      2@      3@      *@      *@      0@      @      $@      $@      @      @      @      @       @      @       @      @       @      �?      �?       @              �?      �?              �?              �?      �?              �?        �]���2      ����	M�����A-*�e

mean squared error��!=

	r-squared(+2>
�L
states*�L	   @���   ���@    ��NA!��^�j2�@)������A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �c@     H�@     �v@     �{@     ȅ@     �@     0�@     x�@     �@     ~�@     �@    ���@    �R�@     V�@     ��@    ���@    ���@    �@�@     ��@    �H�@     �@     ��@    ���@     2�@    @��@    @��@    �p�@    �f�@    @��@    @S�@    �9�@    ��@     7�@    ���@    @7�@    ���@    ���@     ��@    ���@    @��@    @L�@     ��@    ���@    ���@     �@     e�@     M�@    �)�@    ���@    ��@     H�@     ��@    @��@    @��@     ��@    ���@     7�@     j�@    ���@    ���@    �A�@    @��@    @��@    ���@     ��@    @��@    �"�@    ���@     K�@     7�@     +�@     %�@     5�@     q�@     ��@    �Q�@     ��@    ��@     ڽ@     z�@     ��@     ��@     p�@     ��@     �@     ۱@     v�@     2�@     ��@     �@     Ш@     (�@     4�@     ��@     ��@     ��@     ��@     ��@     ��@     P�@      �@     ��@     ��@     ��@     ��@     `�@     ̒@     ��@     4�@     P�@     (�@      �@     Љ@     8�@     ��@     �@      �@     ��@     ��@     0�@     h�@     ��@     �@     ؁@     ��@     X�@     �@      @     p}@     `{@     `|@     `z@     �z@     �z@     �v@     w@     �w@     Pt@     @t@     �t@     Ps@     q@     �r@     @r@     �s@      r@     �s@      q@     �p@     �o@     �k@     @m@     �l@      m@     �h@     `k@     �g@     @h@     `h@     @f@     �g@     @d@     �i@     �b@     �b@     @b@      d@     �`@      c@     `c@     @a@      b@     @a@     �b@      Z@      ^@     �^@     @Z@     �^@     �\@     �]@     @W@     @U@     �[@     �X@      W@      Z@     �V@      S@      S@     �R@     @R@      P@      S@     �S@     �Q@      N@     @S@     @P@     �O@     �N@     �Q@     @R@      N@      M@     �Q@      E@     �J@     �E@     �H@      H@      K@      J@      G@     �F@      I@     �D@      B@      E@      E@      B@     �C@      A@     �B@      C@      B@      ;@     �@@      ?@     �@@     �@@      B@      =@     �B@      5@      ;@      6@      ;@      @@      3@      B@      3@      :@      3@      4@      6@      1@      1@      7@      *@      1@      (@      2@      7@      3@      ;@      2@      (@      *@      $@      (@      *@      9@      6@      5@      9@      .@      0@      $@       @      ,@      @      .@      $@      @      "@      $@      "@      &@      @      "@      @      @      @       @      @      "@      (@      "@       @      @      @      *@      @       @      @      @      "@      @      "@      @      @     �w@     P�@      ,@      $@      $@      @      &@      @      @      $@      (@      "@      "@      @      *@      @      &@      *@      @      ,@      .@      "@      *@      ,@      @      0@      0@      0@      .@      &@      7@      1@      "@      6@      1@      1@      8@      2@      7@      ;@      2@      8@      8@     �@@      @@      ;@      .@      3@      8@      8@      ?@      8@      A@      3@      1@     �@@      >@     �@@      8@      <@     �A@      :@     �E@      H@      8@     �C@      C@     �E@     �F@     �D@      I@      F@      B@     �H@     �G@      D@      F@     �J@     �J@     �E@      G@      G@     �H@     �Q@      H@      N@      I@     �F@      P@      P@      P@     @Q@      M@     �P@      K@     �Q@     �P@     �P@     �P@     �S@     �S@      V@      W@     �S@     @S@     @X@     �X@      X@     �W@     �Y@      U@     �W@     @Y@      `@      Y@      ]@     �_@     �\@      Z@     �^@      \@      `@     �^@     `a@     �^@     @a@     @`@     @b@     �e@      c@     `d@     �f@     �c@     �h@     �g@     �f@      g@     �h@     @g@     @k@     �i@      m@     �g@      m@     @h@     �m@     `n@     �n@     �j@     `o@     �n@     �o@     �r@     �p@     �q@     r@      r@     �u@     Pu@     �t@     pt@     s@     �u@     w@      z@     y@     �w@     @z@     �z@     �y@      @     p@     �}@     �~@     `@     ��@     `�@     X�@     ��@     8�@     Ѓ@     ��@     ��@     �@     X�@     ��@     `�@      �@     x�@     ��@     ��@     P�@     �@     ��@     ��@     ̑@     �@     �@     ��@     ؖ@     l�@     \�@     ��@     �@     |�@     ��@     j�@     ¡@     ��@     ��@     �@     ֦@     ��@     �@     ƪ@     :�@     	�@     K�@     ޲@     H�@     y�@     ��@     p�@     ��@     ��@     ��@    ���@    ���@    � �@     ��@     ��@    �p�@    ���@     ��@    �c�@     ��@    �\�@    �	�@     ��@    ���@     ��@    �7�@    ���@    ���@    ���@    ��@     ��@    @��@     ��@    ���@    ���@    ��@    @2�@    ���@     �@     ��@    �Y�@     Y�@     $�@    `h�@    �e�@    �/�@    ���@    ��@     �@    �K�@    @+�@    �D�@     T�@    ���@     ��@    ���@    ���@    ���@    �+�@    `�@    �Z�@     ��@     ��@     ��@     E�@     ��@    ���@    �v�@    @��@    ���@     ��@    ���@    @�@     ��@     M�@    �s�@    ���@     �@     =�@     3�@     ��@     t�@     ,�@     ܖ@     `�@     X�@     �z@     x�@        
�
predictions*�	   �Byÿ   �5@     ί@!  ��=�?)$�xn�8@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x����[���FF�G �pz�w�7��})�l a�>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�6v��@h�5�@�������:�              �?      �?              @      @      &@      2@      4@      D@     �H@      E@      O@     �U@     @R@     @S@     �V@      V@     @Z@      Z@     @_@     �X@      \@     �X@     @W@     �U@     �W@     @R@     �R@     @Q@     �M@     �N@     �K@      I@     �F@     �E@     �D@      @@     �@@      8@      7@      5@      4@      $@      .@      (@       @      *@      &@      &@      @      ,@      "@      @      &@      @      @      @      �?      @      @      @              @       @       @       @       @              @              @       @      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              @      �?      �?      @              @       @              @       @       @      @      @      @      @      @      @      "@      "@      $@      @      $@      @      $@      $@      @      *@       @      $@      3@      .@      0@      1@      :@      9@      :@      >@      :@      8@      6@      A@     �E@      G@     �D@      D@      B@      E@      E@     �I@     �B@     �F@     �N@      H@      J@     �D@     �D@      D@     �E@      E@     �E@     �D@     �@@      9@      5@      4@      2@      *@      1@      (@      .@      *@      $@      &@       @      @       @      @      @      @      @       @      @      @              �?      �?      �?              �?              �?      �?              �?        �"p123      ��	������A.*�f

mean squared error6�=

	r-squared��>>
�L
states*�L	   �J��    ^�@    ��NA!�������@)���U�5A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �m@     |@     �x@     @     x�@     ��@     ��@     p�@     .�@     n�@     �@    �C�@    �)�@     O�@    ��@    @��@    @��@     W�@     A�@     ��@    ���@    ���@    ���@    ��@     ��@    �q�@      �@     i�@    ���@    ��@    �:�@    �+�@    @~�@    ���@    � �@    ���@    ���@     9�@     :�@    @+�@     #�@    �k�@    @��@     b�@     ��@    ``�@     ��@     ��@     Q�@    ���@    ���@    ��@    ���@    `��@    ���@     ��@    �l�@     )�@    �9�@    �5�@    @S�@    ���@    ���@     ��@    ���@    �r�@    �6�@    ���@    @.�@    ��@     �@     ��@    �Y�@     ��@    �#�@    �}�@     ��@    ��@     ��@     ��@     �@     C�@     h�@     �@     ;�@     ��@     ��@     ¯@     �@     Ы@     ��@     ��@     �@     X�@     (�@      �@      �@     ğ@     4�@     ��@     ��@     �@     T�@     P�@     ��@     ԓ@     Ԓ@     ��@     �@     `�@     (�@     @�@     H�@     ��@     x�@     Ј@     ؈@     ��@     �@     Ѓ@     ��@     ��@     �@     ��@     ��@     ��@     p�@     }@     �@     �}@     {@     @|@     �v@      x@     �u@     �w@      u@      v@     �u@     �t@     `s@     �r@     0s@     s@     �r@     0q@     p@     �q@     `l@     �l@      o@     �k@     �k@     @l@      l@     �i@     @i@     `j@      i@     �j@     @g@     `h@     �e@      f@      d@     @c@      e@     `c@     �c@      c@      ]@     �^@     �`@      ^@     �^@     @_@     �]@      ]@     @[@     �Y@      Y@     �]@     �X@     �Z@     @Y@     �T@     @Z@      T@     �S@      U@     �X@      R@      U@     �V@     @S@     �R@     �N@      N@     �K@      N@     �L@      F@     �J@      Q@      Q@     �R@     �N@      N@      J@     �G@     �H@      K@      E@     �I@     �J@     �D@     �E@      D@     �K@      B@     �@@     �C@      H@      :@      4@     �C@      C@      A@      ;@      <@     �D@     �C@      <@      5@      ;@      >@      B@      4@     �B@      ;@      ;@      2@      3@      A@      9@     �@@      1@      2@      7@      *@      5@      4@      0@      2@      0@      *@      1@      .@      2@       @      0@      3@      *@      $@      "@      ,@      ,@      4@      (@      (@       @      2@      &@      2@      "@      @      $@      "@      (@      "@      (@       @      @      @      $@      "@      @      @      @      @      $@      @      $@      $@      @       @      @      @      @      @      @      @      z@     x�@      (@      ,@      @      &@      ,@      $@      "@      $@      &@      *@      0@      "@      &@      ,@      $@       @      $@      *@      &@      4@      &@      .@      1@      1@       @      1@      1@      5@      6@      3@      2@      6@      3@      .@      4@      ;@      ;@      9@      7@      3@      @@      4@      ;@      6@      4@      >@      8@      ?@      A@      >@      =@      =@      ;@      B@      ;@      >@     �B@      <@      :@      >@     �H@      E@     �E@      D@      F@      C@     �B@      E@      C@     �C@     �B@     �J@     �E@      E@     �J@     �H@     �I@      K@     �P@      G@     �F@     �L@     �E@     �G@      P@      O@      Q@     �Q@      P@      L@     @P@     �R@     �P@      M@     �T@     �U@      S@      V@      O@     �R@     @U@      V@      T@     �T@      U@      X@      U@      [@     �Y@      Z@      X@     @Z@     �\@     �`@      `@      `@      Z@     @]@      [@      `@      `@      `@     �a@     @a@     @f@     �c@     @c@     @d@      d@     @c@     �c@     @f@     �d@     �d@     @g@      j@     `f@     `j@     @f@     �h@      m@     @i@     �l@     �l@     �l@     �m@     @l@      l@     Pr@     0q@     pq@      n@      r@     �r@     @q@     @s@     0s@     0t@      t@     �v@      y@     �w@     �w@     �y@     �y@     x@     `{@     0{@     @z@     �{@     �@     �@     0@     �|@     �@     (�@     Ё@     @�@     p�@     `�@     8�@     H�@     `�@     p�@     H�@     ؇@     8�@      �@     Ѝ@     Ȏ@     ؏@     ��@      �@     (�@     �@     �@     p�@     ��@     ��@     ��@     ̚@     ��@     ��@     z�@      �@     ��@     N�@     ��@     :�@     ��@     J�@     L�@     6�@     ԯ@     װ@     �@     ��@     Ѵ@     %�@     ��@     C�@     N�@     Ⱦ@     ��@     &�@     �@     A�@     "�@    ���@     ��@    ���@     ��@     V�@    ���@    ���@    ���@    @��@    �s�@    @^�@     ��@     ��@    `$�@    �r�@    �{�@    ���@    �<�@    �O�@     #�@    @2�@    @��@     ��@    �i�@    �A�@    ���@    ��@     ��@    �G�@    @��@    `��@     I�@    �*�@    @�@    �<�@    �
�@    @��@    `D�@    ���@    ���@    �N�@    @��@    ���@    ���@    �)�@    `G�@    ���@    @K�@     M�@    �1�@    @��@    ���@     ��@    ���@    ���@    ���@    �h�@     �@     x�@    @L�@    �7�@    @�@    �S�@     6�@     '�@     i�@     �@     ��@     ��@     $�@     �@     ��@      }@     x�@        
�
predictions*�	   ��t��   �D`@     ί@!  ����H@)gkJ�M�@@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��})�l a��ߊ4F�𾹍�?�ګ>����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>�FF�G ?��[�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?3?��|�?�E̟���?S�Fi��?ܔ�.�u�?�DK��@{2�.��@�������:�              @      @      @      @      (@      0@      5@     �@@     �C@      H@     �A@      M@     �K@     �G@      P@     �P@      I@      N@      M@      I@     �K@      L@      N@     �J@      F@      K@     �H@     �A@      F@     �D@      C@     �C@      @@      ?@      1@      6@      0@      3@      6@      8@      .@      2@      2@      &@      1@      @      "@      @      @      @      @      @      @      @      @      �?      @      @      @      @      @       @       @      �?       @              @              �?      �?      �?       @      �?              �?               @       @      �?              �?      �?              �?              �?              �?              �?      �?      �?              �?              �?       @              �?              �?      �?      @              @      @      �?       @       @      @      @       @      @      @      @      @      @      @      @      @       @      "@      @      @      $@      "@      ,@      .@      (@      2@      2@      1@      *@      >@      2@      :@      0@     �F@      B@     �D@     �H@      E@      P@     �C@     �S@     �M@      V@     @T@     �V@      U@      Z@     @Z@     @V@     �Z@     @Z@     �U@     �S@     �P@     �P@     @P@     �I@     �D@     �C@      B@      >@      ?@      6@      .@      0@      1@       @      3@      &@      $@      &@      "@       @      @               @      @      @      �?      @      �?      @      @      @               @      �?              �?              �?              �?        �ݿ�3      _  	ݳ����A/*�g

mean squared error�"=

	r-squaredh�1>
�L
states*�L	   �Q��   @j�@    ��NA!�?꒟�@)���]TA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             s@     �{@     �u@     �@     ��@     ܖ@     D�@     Ġ@     ¥@     2�@     ��@     �@    ���@     ��@    �F�@    @m�@     �@     2�@     ��@     �@    ���@     [�@     F�@    ���@     Q�@    �w�@    ���@    ���@    ��@    @7�@    �c�@    @��@     ��@    ���@     m�@    ���@     O�@    �}�@     *�@     �@    �9�@    ���@    @5�@    ���@    �
�@    ���@    @��@    `��@    ���@     �@    `��@    `t�@     ��@     ��@    ���@    ���@     x�@     4�@    @1�@    ���@    @�@     9�@    �{�@    @��@    �E�@    @��@     k�@     ��@    ���@    ���@    ���@     ��@    ���@    ��@    �n�@    �
�@     ��@     ��@     d�@     �@     _�@     %�@     ^�@     -�@     ;�@     ��@     ��@      �@     �@     ��@     ��@     �@     ��@     2�@     :�@     V�@     �@     �@     $�@     8�@     ��@     l�@     Ԗ@     0�@     Ĕ@     h�@     �@     ��@     <�@     ��@     @�@     Ѝ@     (�@     p�@     p�@     ��@     ��@     `�@     ��@     ��@     P�@     �@     Ё@     ��@     h�@     h�@     �~@     �z@     �}@     �{@     �z@     �z@     �x@     �x@     �x@     v@     `w@     `w@     �u@     �u@      v@     �r@     �q@     Pt@     r@     0p@     �n@     �p@      l@     �q@     �i@      p@     @l@     �k@     @k@     �h@     @l@     `l@     `j@     �g@     `f@     �g@      g@     �e@     �c@     @c@     �c@     �a@     `c@      d@     �`@      b@     �\@     `b@     `a@     �a@      \@     �_@      [@     @]@     @Z@     �U@     �T@     �V@      U@      \@     �W@     �Y@     �S@     @R@     @T@      V@      V@     �P@     �P@     @R@     @P@     �T@     �R@     �O@      P@     �U@      L@      L@     �M@     �E@     �E@     �J@     @P@     �I@      N@      K@     �G@     �E@     �F@      K@     �F@     �E@      >@     �E@      =@      B@      B@      H@     �B@     �D@      @@     �A@     �D@     �E@      ?@      6@     �@@      =@      ?@      9@      5@      0@      >@      8@      6@      4@      9@      0@      6@      5@      6@      6@      0@      1@      .@      .@      1@      4@      *@      .@      1@      4@      4@      3@      0@      1@      &@      1@      *@      *@      (@      *@      $@      "@      .@      @      @      .@      $@       @      $@      (@      ,@       @      $@      .@      $@      @       @       @      "@      $@      *@      "@      @      @      @      @      @      �?       @      @      $@      @      "@      @     `z@     8�@       @      @      (@      &@       @      $@      @      $@      1@      2@      "@      @      0@      ,@      ,@      "@      .@      $@      1@      &@      1@      $@      &@      *@      5@      0@      3@      "@      1@      *@      3@      2@      ,@      3@      .@      <@      ?@      0@      2@      .@      0@      0@      <@     �A@      :@      6@      6@      B@      ;@      6@      <@      ;@      7@      :@      9@      B@      >@     �A@      A@      D@      C@      @@     �E@      B@     �@@     �F@      A@     �C@      I@     �D@     �H@     �E@     �L@      D@      B@      ?@     �C@      J@     �G@     �O@     �I@     �K@     �G@     �L@      Q@     @Q@      E@      N@     �K@     �O@     �T@     �L@     �R@     �Q@      T@     �Q@     �Q@     @S@     �T@     @W@     �U@     �S@     �T@      W@     @V@      X@     @V@     @]@     �\@      V@      Z@      [@     @W@     �X@     �Z@     �W@     �[@     �`@     �`@      \@     �]@     �c@     �`@     �a@     �a@     �c@     �c@     �b@     `c@      b@     �d@     �e@     @d@     `e@     �e@     @f@     `f@      f@     �k@     @h@      i@      j@      k@     �k@     �o@      l@      q@     �m@     @o@     �p@     �p@     �q@     �p@     Pr@     �q@     �s@     �s@     �s@     Pv@     `v@     pu@     �s@     `w@      y@     y@     �y@     |@     �{@     �}@     `~@     �|@     �|@     `�@     p�@     ��@     P�@     ��@     �@     �@     ��@     �@     p�@     ��@     ��@     H�@     �@     �@     ��@     ��@     ��@     ؏@     t�@     ȑ@     ��@     �@     ��@     �@     ��@     �@     ��@     �@     �@     ��@     ğ@     ��@     >�@     R�@     Σ@     ��@     T�@      �@     ��@     ά@     
�@     #�@     v�@     ϲ@     ;�@     ��@     &�@     ��@     '�@     �@     
�@     ��@    ���@     :�@    ��@    ���@    �A�@     I�@    ���@    ���@    @2�@     ��@    @A�@    �	�@    @�@    ���@    @�@    ���@     #�@     I�@    ���@    ���@    ���@    `)�@    `Q�@    �7�@     <�@    ���@    �.�@    ��@    �l�@    ��@     ��@    �p�@    ���@    ���@    @��@    `|�@    �Q�@    ��@     �@    ��@    �?�@    �T�@    `��@    �]�@     �@    ���@    @D�@    @��@     ��@    �-�@    ���@     O�@    �A�@    ��@    ���@     ��@    ���@     T�@     ��@     `�@    ��@     ��@    @��@    @u�@    ���@     r�@     x�@     ^�@     &�@     ,�@     �@     ��@     Ė@     ��@      �@     �|@     �@        
�
predictions*�	   ��cǿ    �	@     ί@!  ȊX�,�) ��t�S7@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�1��a˲���[���FF�G �>�?�s����ߊ4F��h���`�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾���%�>�uE����>�h���`�>�ߊ4F��>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?W�i�b�?��Z%��?�1%�?\l�9�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?h�5�@�Š)U	@�������:�              �?      �?      �?      @       @      &@      "@      ?@      @@      C@      I@      M@      K@      J@     @Q@      R@     �Q@     �T@      V@     �V@      S@     �T@     �V@     �W@     @U@     @T@      S@     �R@     �T@     @R@      T@     �K@     �I@     �H@     �D@      C@     �B@     �C@      ;@      D@      ?@      9@      4@      1@      3@      0@      *@      1@      0@      &@      ,@      "@      3@      @      $@      @      @       @      @      @      @       @       @               @       @       @      @      @       @       @      �?      �?      @      @              @              �?               @       @              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?               @               @       @       @      @              �?      �?              @      �?      @       @      @       @      @      @      @      "@      @      @      "@      @      @      (@      $@      "@      *@      $@      4@      4@      ,@      .@      7@      6@      ?@      @@     �@@      @@     �A@      E@     �E@     �E@      M@     �L@     �M@      R@     �I@     �G@      I@     �M@      L@     �L@      L@      L@     �K@      E@      A@      8@      ?@      ;@      4@     �A@      7@      2@      .@      (@      "@       @      .@      @      &@      @      @      "@      @      @       @              @      �?       @               @               @              �?               @      �?              �?              �?        ��9$"3      ����	9G����A0*�f

mean squared error!=

	r-squared(7>
�L
states*�L	   ���   ���@    ��NA!�Vܾ��@)	��ArA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             @u@     �x@     `v@     (�@     ��@     ��@     ��@     z�@     ��@     Ь@     f�@     �@    @5�@    @��@    ���@    ���@    ���@     �@    ���@    ���@     )�@    �m�@     �@    @��@    ���@    @��@    ���@    ���@    �i�@     F�@     ��@    @��@    ���@    ���@    ���@    ���@     \�@    ���@    �-�@     ��@     ��@      �@    ���@    @A�@     ��@    ���@    ���@    �7�@    ��@    ���@     x�@    @J�@    `��@    ���@     ��@    ���@    `��@    �.�@    ���@    @��@     :�@    ���@     ��@    �^�@    @��@     I�@    @��@    ���@    ���@     _�@    �]�@     ��@     ^�@     ��@    ���@    �F�@     �@     ��@     �@     ��@     |�@     ׸@     Y�@     �@     �@     б@     �@     ��@     |�@     B�@     ��@     J�@     p�@     j�@     ��@     ��@     H�@     �@     �@     �@     ԛ@     x�@     �@     ��@     ��@     �@     ��@     ��@     ��@     p�@     ��@     �@     8�@     @�@     @�@     �@     (�@     ��@     h�@     ��@     Ѓ@     ��@     h�@     ��@     ��@      �@     �@     �@     @~@     �{@     P}@     �~@     �y@      z@     �w@     @u@     Pv@     �v@     �v@     @r@     �s@     �s@     �s@     �p@     �s@      o@     r@     �p@     �p@      p@      o@     `p@     �j@     @j@     @m@     �j@      i@     �i@     @h@     �h@     �g@     �e@     @e@     �c@     �c@      c@      c@      c@      c@     �a@     �b@     `a@     �\@     �Y@     �c@     �_@     @\@     �\@     @Y@     �X@     �Y@     �V@      V@     �Y@     @Z@     �U@     �T@     @Q@     �U@     �S@     @W@      O@     �S@     @R@     �Q@      O@     �U@     @W@      R@     �P@     �P@     �O@     �P@      P@      K@      M@     �Q@     �I@      O@      E@     �I@      H@     �D@     �K@      J@     �E@      @@      L@     �G@     �E@      D@     �C@     �@@     �D@      ?@     �@@      A@      5@      @@      D@      >@     �A@      <@      C@      >@      A@      7@      >@      3@      2@     �E@      6@      A@      7@      ;@      .@      =@      1@      ,@      1@      =@      .@      7@      1@      3@      1@      4@      8@      .@      4@      1@      "@      (@       @      (@      &@      3@      .@      &@      4@       @      *@      *@      @      @      $@      &@      (@      $@      $@      @      ,@      @       @      @      @       @      @       @      @      @      *@      @      &@      @      @      @      "@      @      "@      @      $@     Pz@     `�@      "@      @      &@       @      @      @      *@      "@      $@      @      $@      (@      @      0@      .@      @      $@      .@      .@      $@      (@      .@      3@      (@      $@      .@      .@      (@      (@      1@      4@      1@      7@      3@      ,@      5@      9@      3@      6@      &@      8@      9@      >@      ;@      .@      6@      =@      ?@      =@      7@     �A@      =@      B@      5@      8@      @@      B@      :@      E@     �C@      B@      D@      D@      C@      >@     �@@      >@      >@     �F@     �A@     �@@     �H@     �@@      G@      F@     �H@      J@     �F@     �K@     �H@     �J@      H@      M@      K@     �N@     �J@      O@      J@     @S@     @R@      Q@     �R@     �P@     �S@      O@      S@     @R@     @R@     �R@     �T@     �R@      T@     @W@     �W@     �V@     �U@     �U@     �X@      X@      W@     @Z@     @`@     @[@     @Y@     �^@     �^@     �]@      c@      a@     �`@     �c@     �`@      a@     �`@     �b@     `b@     �]@     �`@     �c@      c@     �g@      e@     �g@     �g@      d@     `c@     �g@     @i@      k@     @l@      i@     �m@     `i@      n@      n@     @l@     �m@      n@      n@     �p@     �p@     �p@     @s@     �q@     �q@     �r@     �s@     t@     �u@     �u@     @w@     �w@     �x@     �x@      w@     �y@     �{@     p{@     p|@      }@     `}@     ��@     p@     ��@     ��@     ��@     H�@     0�@     0�@     ؃@     ��@     ��@     �@     `�@     �@     ��@      �@     ��@     X�@     t�@     Ԑ@     ��@     В@     �@     ��@     ��@     �@     ܖ@     З@     ��@     ��@     ��@     ��@     ��@     ��@     J�@     b�@     v�@     `�@     ̧@     �@     ƪ@     V�@     ��@     ��@     ��@     ײ@     m�@     �@     f�@     ��@     �@      �@    �9�@    ���@     `�@    ���@    ���@    ��@     �@    �f�@    ��@    @m�@    ���@     K�@     r�@    ���@     ��@     �@    �B�@    `e�@    ���@     ��@     ��@    ��@    @��@    ���@    ���@    �-�@    ���@    ���@     ]�@    @��@    �*�@    �T�@     =�@     ��@    ���@    �w�@    �^�@    ���@    ���@    @V�@    �2�@    �(�@     #�@    �^�@    `��@     ~�@     �@    �\�@    �O�@    ���@    `��@    @��@    �l�@     x�@     y�@    ���@    ���@    @
�@    ���@    @��@    �1�@    ���@    ���@    @��@    @��@    ���@     $�@    ���@     ̺@     R�@     ��@     ©@     ��@     ��@     (�@     (�@     ��@      {@     �@        
�
predictions*�	    $�˿   @�B@     ί@!  ��5�1@)1��\�B@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9��x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G ����%ᾙѩ�-߾
�/eq
Ⱦ����ž8K�ߝ�>�h���`�>��Zr[v�>O�ʗ��>��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�DK��@{2�.��@�������:�              �?               @               @      @      6@      7@      1@      ?@      A@     �C@     �G@      J@     @T@     �Q@      Q@      P@      L@     �I@     �L@     �P@      U@     @P@     �G@     �H@      I@     �P@     �J@     @Q@     �K@     �I@     �A@     �@@      E@     �@@      E@      9@      @@      ;@      5@      8@      >@      1@      2@      0@      0@      (@      "@      &@      @      &@      "@      &@       @      @      @      @      @      @      �?      @      @      @      @      @       @      @               @      �?      �?      �?       @               @              �?      �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?              �?               @      �?      @      @      @       @      @      �?       @      @      @       @      @      @      @      @      (@      (@      @      (@      &@      (@       @      @      "@      0@      *@      &@      :@      :@      8@      ;@      @@      =@     �A@      >@     �I@     �E@      K@     �K@     �P@     @R@     �R@      R@     @S@     @S@      W@     �X@      U@      U@     @W@     �S@     �S@     �M@     �K@      H@      H@      A@      A@      <@      4@      3@      5@      .@      *@      $@       @      (@       @      @      @      @       @      @      @       @       @      @       @               @      �?      �?              �?              �?      �?              �?              �?              �?              �?        �	��2      ��ٞ	�b�����A1*�d

mean squared error��=

	r-squared��>>
�L
states*�L	   �0��   @q�@    ��NA!C���@)@�}���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             py@     Pv@     @z@     ��@     8�@     ܖ@     �@     �@     .�@     ܬ@     �@     ��@     /�@    �O�@    ���@     c�@    ���@     "�@     ��@     E�@    �$�@    ���@    ���@    ���@     &�@    @��@     ��@     ��@    @M�@    � �@    �I�@    ���@    ���@    ���@    ���@    ���@     w�@     ��@    ���@    ���@    @��@     ��@    �O�@     ��@    `��@    �E�@    @c�@    �{�@    �k�@    �&�@    �%�@    ���@    @��@    ���@    �{�@     \�@     )�@    @��@    @��@    �k�@    �1�@    ���@    ���@     ��@    �u�@    ���@     x�@    �S�@    ��@    @�@     �@    ���@     ��@    ���@     ~�@    ���@     ��@     v�@     I�@     9�@     ݺ@     �@     z�@     $�@     
�@     ^�@     ��@     ��@     ��@     ��@     6�@     (�@     |�@     N�@     >�@     ��@     ��@     ��@     ԟ@     H�@     ��@     �@     ��@     �@     P�@     P�@     ��@     X�@     <�@     ��@     ��@     ��@     ��@     ��@     Њ@     ��@     ��@     ȇ@      �@     p�@     �@     ��@     �@     H�@     ��@     Ё@     ��@     ��@     p�@      @     �|@      }@     �w@     z@     Pz@     �v@     �v@     �v@     �w@      w@     �s@      t@     Pu@     �t@     r@     �q@     @s@     pp@     �p@     �p@     Pp@      n@      p@     �n@     `i@      k@     �n@     �l@      k@      h@     �i@     �d@      k@     �g@      c@      e@      e@     �b@      d@     �b@      a@     �`@     �`@      _@     �a@     �a@     �W@      ]@     �`@     �]@     @Z@     �\@     �X@     �X@      X@     �R@     @Y@     @U@     �Y@     @V@     �V@     �V@      S@     @T@     �R@     @T@     �P@     �S@     �R@     @R@      P@      P@     �I@      Q@     �N@      L@      K@     �K@      M@      F@      I@      O@     �D@     �J@     �G@     �G@      D@      D@     �E@     �F@      G@     �@@     �@@     �C@     �H@     �B@     �E@      :@     �A@     �@@      =@      @@      :@      A@      E@      8@      ;@      ;@      5@      9@      ;@      <@      ?@      7@      5@      8@      .@      6@      3@      2@      5@      4@      0@      (@      *@      1@      (@      .@      $@      $@      4@      5@       @      @      &@      0@      &@      .@      $@      "@      0@      *@      *@      *@      *@      @      0@      &@      &@      (@       @      *@      "@      *@      $@      @      @       @      @      @      @      "@      @      @      (@      &@      $@      @      @      @      $@      *@     0z@     H�@       @      @      $@      &@       @       @      @      $@       @      .@      "@      &@      &@      &@      (@      3@      (@      $@      &@      *@       @      0@      $@      ,@      &@      1@      *@      3@      2@      3@      5@      3@      9@      2@      5@      *@      ,@      1@      ,@      9@      1@      5@      3@      ?@      <@      9@      =@      @@      ?@      9@      <@      :@      >@     �@@      =@      A@      ;@      =@      ?@      ;@      C@      C@      =@     �G@     �F@      7@      E@     �F@      A@      C@     �E@     �H@      G@     �C@     �G@     �O@     �M@     �L@     �H@      N@     �O@     �Q@     �M@      O@      N@     �P@     �M@      N@     �M@     @P@     @S@     �T@      R@     �Q@     @U@      S@     @V@     @U@     @U@      U@     �T@     �V@     �T@     �S@     @V@     @X@     �W@     @U@      W@      Y@      Y@     �\@      \@     �Z@     �X@     �^@      `@      [@     �]@     �]@      a@      a@     �a@     @e@     �d@     @^@     �d@     `e@     �b@     �b@      e@     �g@     �d@     `g@     �i@     �f@     �k@     �h@     �g@     �h@     @l@     �i@      l@     @m@     �o@     `n@     Pp@     �n@     p@     pq@     pr@     `r@     pr@     �s@     �r@      s@     Ps@     �t@     �u@     �v@     `v@     �z@     �y@     �y@     @z@     �{@     �{@      {@     `�@     0@     @}@      ~@      �@     ��@     �@     ��@     0�@     �@     p�@     ��@     �@      �@     ��@     ��@     h�@     ��@     p�@     H�@     �@     ��@     4�@     ԑ@     x�@     t�@     ̓@     ��@     |�@     �@     ��@     ̙@     ��@     ܜ@      �@     <�@     ��@     *�@     z�@     ĥ@     ��@     X�@     ��@     ��@     8�@     Ԯ@     ��@     ��@     ]�@     δ@     ��@     �@     J�@     ��@     ݿ@    �X�@    ���@     �@     ��@    �J�@    �B�@    ���@    � �@    ���@    �`�@    �I�@    @x�@    �P�@    �k�@    @a�@    �c�@    @��@    ��@     ��@    ���@    �h�@    `-�@    ���@    ���@    ���@    �\�@    @��@    @�@    �_�@    `��@     ��@    ���@    `l�@     �@    ���@    ���@    ���@    �C�@    ���@     ��@    @i�@    ���@    @��@     b�@    @��@     W�@     )�@    `s�@    `C�@    �W�@    �6�@    �L�@    ���@    ��@    @�@    ���@    ���@    ���@    ���@    ���@    ���@     ��@     �@    ��@    ���@    ���@     ��@    ���@     o�@     =�@     ذ@     �@     ĥ@     ޢ@     �@     Ѝ@     ��@     P�@     ��@        
�
predictions*�	   �cӿ   ��@     ί@!  �]A@)�@w�[�@@2��Ca�G�Կ_&A�o�ҿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ�>h�'��f�ʜ�7
�O�ʗ��>>�?�s��>�FF�G ?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�6v��@h�5�@�DK��@{2�.��@�������:�              �?              �?      @      @      &@      $@      &@      1@      6@      @@      C@      B@     �H@     �F@      N@      J@     �I@     �K@     �P@     �O@     �M@     �L@      O@     �M@     �P@     �L@     �F@      B@     �B@      D@     �C@      D@      C@      A@      :@      8@      ;@      =@      :@      *@      (@      &@      2@      2@      (@      (@      &@      2@      @      &@      "@      @      "@      @       @      @      @      @      @       @      �?      �?       @      @              @       @       @       @      �?       @      �?      �?       @              �?      �?               @              �?      �?              �?              �?              �?              �?      @      �?       @      �?      �?               @               @      @       @      @      @      @      @      @      @       @      @      @       @      "@      &@      @      "@      @      &@      &@      &@      4@      0@      3@      2@      ?@      @@      <@     �A@      @@      C@      B@     �D@      L@     �H@     �Q@     @R@      U@     �S@     �Q@     �T@      Y@     @[@     �\@     �Z@     @\@     �V@      R@     �V@     @Q@      T@     �P@     �N@     �D@     �A@     �A@      @@      .@      2@      &@      *@      .@      0@      @      @      @      @      @      @              �?       @      �?               @              �?      �?      �?              �?              �?              �?              �?        y��K