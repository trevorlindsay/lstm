       �K"	  �z���Abrain.Event:296v�v�     |���	F��z���A"��
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
N
model/pack/1Const*
dtype0*
value	B :.*
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
d
model/zerosFill
model/packmodel/zeros/Const*
T0*'
_output_shapes
:���������.
P
model/pack_1/1Const*
dtype0*
value	B :.*
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
j
model/zeros_1Fillmodel/pack_1model/zeros_1/Const*
T0*'
_output_shapes
:���������.
P
model/pack_2/1Const*
dtype0*
value	B :.*
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
j
model/zeros_2Fillmodel/pack_2model/zeros_2/Const*
T0*'
_output_shapes
:���������.
P
model/pack_3/1Const*
dtype0*
value	B :.*
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
j
model/zeros_3Fillmodel/pack_3model/zeros_3/Const*
T0*'
_output_shapes
:���������.
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
 *��Z?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"�   �   *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *ZN�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *ZN<*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
|
:model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/splitSplit:model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dim.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add*
	num_split*
T0*@
_output_shapes.
,:	�.:	�.:	�.:	�.
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
_output_shapes
:	�.
�
2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�.
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel/zeros2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�.
�
/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�.
�
1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�.
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�.
s
.model/RNN/MultiRNNCell/Cell0/dropout/keep_probConst*
dtype0*
valueB
 *��Z?*
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
_output_shapes
:	�.
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell0/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/sub*
T0*
_output_shapes
:	�.
�
3model/RNN/MultiRNNCell/Cell0/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
:	�.
�
(model/RNN/MultiRNNCell/Cell0/dropout/addAdd.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob3model/RNN/MultiRNNCell/Cell0/dropout/random_uniform*
T0*
_output_shapes
:	�.
�
*model/RNN/MultiRNNCell/Cell0/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell0/dropout/add*
T0*
_output_shapes
:	�.
�
(model/RNN/MultiRNNCell/Cell0/dropout/InvInv.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell0/dropout/mulMul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0*
_output_shapes
:	�.
�
*model/RNN/MultiRNNCell/Cell0/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mul*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0*
_output_shapes
:	�.
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:	\�*
	container *
shared_name *
_output_shapes
:	\�
�
Ymodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB"\   �   *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *ZN�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *ZN<*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape*
_output_shapes
:	\�*
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
_output_shapes
:	\�
�
Smodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
:	\�
�
?model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0*
_output_shapes
:	\�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
:	\�
�
Cmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatConcatCmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dim*model/RNN/MultiRNNCell/Cell0/dropout/mul_1model/zeros_3*
_output_shapes
:	�\*
T0*
N
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
|
:model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/splitSplit:model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dim.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add*
	num_split*
T0*@
_output_shapes.
,:	�.:	�.:	�.:	�.
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
_output_shapes
:	�.
�
2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�.
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel/zeros_22model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�.
�
/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�.
�
1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�.
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�.
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�.
s
.model/RNN/MultiRNNCell/Cell1/dropout/keep_probConst*
dtype0*
valueB
 *��Z?*
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
_output_shapes
:	�.
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell1/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/sub*
T0*
_output_shapes
:	�.
�
3model/RNN/MultiRNNCell/Cell1/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
:	�.
�
(model/RNN/MultiRNNCell/Cell1/dropout/addAdd.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob3model/RNN/MultiRNNCell/Cell1/dropout/random_uniform*
T0*
_output_shapes
:	�.
�
*model/RNN/MultiRNNCell/Cell1/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell1/dropout/add*
T0*
_output_shapes
:	�.
�
(model/RNN/MultiRNNCell/Cell1/dropout/InvInv.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell1/dropout/mulMul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0*
_output_shapes
:	�.
�
*model/RNN/MultiRNNCell/Cell1/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0*
_output_shapes
:	�.
Y
model/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
n
model/concatIdentity*model/RNN/MultiRNNCell/Cell1/dropout/mul_1*
T0*
_output_shapes
:	�.
d
model/Reshape/shapeConst*
dtype0*
valueB"����.   *
_output_shapes
:
e
model/ReshapeReshapemodel/concatmodel/Reshape/shape*
T0*
_output_shapes
:	�.

model/dense_wVariable*
dtype0*
shape
:.*
	container *
shared_name *
_output_shapes

:.
�
.model/dense_w/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@model/dense_w*
valueB".      *
_output_shapes
:
�
,model/dense_w/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *ZN�*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *ZN<*
_output_shapes
: 
�
6model/dense_w/Initializer/random_uniform/RandomUniformRandomUniform.model/dense_w/Initializer/random_uniform/shape*
_output_shapes

:.*
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

:.
�
(model/dense_w/Initializer/random_uniformAdd,model/dense_w/Initializer/random_uniform/mul,model/dense_w/Initializer/random_uniform/min* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:.
�
model/dense_w/AssignAssignmodel/dense_w(model/dense_w/Initializer/random_uniform*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes

:.
x
model/dense_w/readIdentitymodel/dense_w* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:.
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
 *ZN�*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *ZN<*
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
t
model/zeros_4Const*
dtype0*&
valueB�.*    *'
_output_shapes
:�.
�
model/VariableVariable*
dtype0*
shape:�.*
	container *
shared_name *'
_output_shapes
:�.
�
model/Variable/AssignAssignmodel/Variablemodel/zeros_4*
validate_shape(*!
_class
loc:@model/Variable*
use_locking(*
T0*'
_output_shapes
:�.
�
model/Variable/readIdentitymodel/Variable*!
_class
loc:@model/Variable*
T0*'
_output_shapes
:�.
�
model/Assign/value/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model/Assign/value/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model/Assign/valuePackmodel/Assign/value/0model/Assign/value/1*'
_output_shapes
:�.*
T0*
N
�
model/AssignAssignmodel/Variablemodel/Assign/value*
validate_shape(*!
_class
loc:@model/Variable*
use_locking( *
T0*'
_output_shapes
:�.
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
_output_shapes
:	�.
�
*model/gradients/model/MatMul_grad/MatMul_1MatMulmodel/Reshape&model/gradients/model/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:.
d
(model/gradients/model/Reshape_grad/ShapeShapemodel/concat*
T0*
_output_shapes
:
�
*model/gradients/model/Reshape_grad/ReshapeReshape(model/gradients/model/MatMul_grad/MatMul(model/gradients/model/Reshape_grad/Shape*
T0*
_output_shapes
:	�.
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
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mulMul*model/gradients/model/Reshape_grad/Reshape*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0*
_output_shapes
:	�.
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/SumSumCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mulUmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/ReshapeReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/SumEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape*
T0*
_output_shapes
:	�.
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/gradients/model/Reshape_grad/Reshape*
T0*
_output_shapes
:	�.
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Sum_1SumEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Wmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape_1ReshapeEmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Sum_1Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/SumSumAmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/ReshapeReshapeAmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/SumCmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Shape*
T0*
_output_shapes
:	�.
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Shape_1*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�.
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Sum_1SumJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mul]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/SumSumGmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mulYmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape*
T0*'
_output_shapes
:���������.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1Mulmodel/zeros_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Sum_1SumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Shape_1*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/subSubMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub/x2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�.
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
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
��*
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
��
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
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0*
_output_shapes
:	�\
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	\�
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
_output_shapes
:	�.
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice_1SliceTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul\model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffset:1Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:���������.
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
_output_shapes
:	�.
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/SumSumCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mulUmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/ReshapeReshapeCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/SumEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape*
T0*
_output_shapes
:	�.
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice*
T0*
_output_shapes
:	�.
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Sum_1SumEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Wmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape_1ReshapeEmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Sum_1Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Amodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/SumSumAmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/ReshapeReshapeAmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/SumCmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Shape*
T0*
_output_shapes
:	�.
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Shape_1*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�.
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Sum_1SumJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mul]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/SumSumGmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mulYmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/ReshapeReshapeGmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/SumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape*
T0*'
_output_shapes
:���������.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1Mulmodel/zerosMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Sum_1SumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1ReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Sum_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Shape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/SumSumImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul[model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
:
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeReshapeImodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Sum_1SumKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1]model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
:
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1ReshapeKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Sum_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Shape_1*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub/xConstN^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1*
dtype0*
valueB
 *  �?*
_output_shapes
: 
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/subSubMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub/x2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�.
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�.
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�.
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
_output_shapes
:	�.
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
��*
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
��
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
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
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
�
model/global_norm/L2Loss_4L2Loss*model/gradients/model/MatMul_grad/MatMul_1*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes
: 
�
model/global_norm/L2Loss_5L2Loss(model/gradients/model/add_grad/Reshape_1*;
_class1
/-loc:@model/gradients/model/add_grad/Reshape_1*
T0*
_output_shapes
: 
�
model/global_norm/packPackmodel/global_norm/L2Lossmodel/global_norm/L2Loss_1model/global_norm/L2Loss_2model/global_norm/L2Loss_3model/global_norm/L2Loss_4model/global_norm/L2Loss_5*
_output_shapes
:*
T0*
N
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
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_2MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_1Identitymodel/clip_by_global_norm/mul_2*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_3MulVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	\�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_2Identitymodel/clip_by_global_norm/mul_3*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0*
_output_shapes
:	\�
�
model/clip_by_global_norm/mul_4MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_3Identitymodel/clip_by_global_norm/mul_4*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_5Mul*model/gradients/model/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:.
�
6model/clip_by_global_norm/model/clip_by_global_norm/_4Identitymodel/clip_by_global_norm/mul_5*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:.
�
model/clip_by_global_norm/mul_6Mul(model/gradients/model/add_grad/Reshape_1model/clip_by_global_norm/mul*;
_class1
/-loc:@model/gradients/model/add_grad/Reshape_1*
T0*
_output_shapes
:
�
6model/clip_by_global_norm/model/clip_by_global_norm/_5Identitymodel/clip_by_global_norm/mul_6*;
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
model/zeros_5Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adammodel/zeros_5*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_6Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Lmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/AssignAssignEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1model/zeros_6*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
\
model/zeros_7Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
shared_name 
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/AssignAssignAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adammodel/zeros_7*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
\
model/zeros_8Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/zeros_8*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
d
model/zeros_9Const*
dtype0*
valueB	\�*    *
_output_shapes
:	\�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container *
_output_shapes
:	\�*
dtype0*
shape:	\�*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adammodel/zeros_9*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0*
_output_shapes
:	\�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
:	\�
e
model/zeros_10Const*
dtype0*
valueB	\�*    *
_output_shapes
:	\�
�
Emodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container *
_output_shapes
:	\�*
dtype0*
shape:	\�*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
shared_name 
�
Lmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/AssignAssignEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/zeros_10*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0*
_output_shapes
:	\�
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0*
_output_shapes
:	\�
]
model/zeros_11Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
shared_name 
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/AssignAssignAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adammodel/zeros_11*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_12Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
shared_name 
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/AssignAssignCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/zeros_12*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
c
model/zeros_13Const*
dtype0*
valueB.*    *
_output_shapes

:.
�
model/model/dense_w/AdamVariable*
	container *
_output_shapes

:.*
dtype0*
shape
:.* 
_class
loc:@model/dense_w*
shared_name 
�
model/model/dense_w/Adam/AssignAssignmodel/model/dense_w/Adammodel/zeros_13*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes

:.
�
model/model/dense_w/Adam/readIdentitymodel/model/dense_w/Adam* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:.
c
model/zeros_14Const*
dtype0*
valueB.*    *
_output_shapes

:.
�
model/model/dense_w/Adam_1Variable*
	container *
_output_shapes

:.*
dtype0*
shape
:.* 
_class
loc:@model/dense_w*
shared_name 
�
!model/model/dense_w/Adam_1/AssignAssignmodel/model/dense_w/Adam_1model/zeros_14*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes

:.
�
model/model/dense_w/Adam_1/readIdentitymodel/model/dense_w/Adam_1* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:.
[
model/zeros_15Const*
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
model/model/dense_b/Adam/AssignAssignmodel/model/dense_b/Adammodel/zeros_15*
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
model/zeros_16Const*
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
!model/model/dense_b/Adam_1/AssignAssignmodel/model/dense_b/Adam_1model/zeros_16*
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
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_2*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0*
_output_shapes
:	\�
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_3*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
)model/Adam/update_model/dense_w/ApplyAdam	ApplyAdammodel/dense_wmodel/model/dense_w/Adammodel/model/dense_w/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_4* 
_class
loc:@model/dense_w*
use_locking( *
T0*
_output_shapes

:.
�
)model/Adam/update_model/dense_b/ApplyAdam	ApplyAdammodel/dense_bmodel/model/dense_b/Adammodel/model/dense_b/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_5* 
_class
loc:@model/dense_b*
use_locking( *
T0*
_output_shapes
:
�
model/Adam/mulMulmodel/beta1_power/readmodel/Adam/beta1U^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam*K
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
�
model/Adam/mul_1Mulmodel/beta2_power/readmodel/Adam/beta2U^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam*K
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
�

model/AdamNoOpU^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdamU^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdamS^model/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam*^model/Adam/update_model/dense_w/ApplyAdam*^model/Adam/update_model/dense_b/ApplyAdam^model/Adam/Assign^model/Adam/Assign_1
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
model/HistogramSummary/values/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model/HistogramSummary/values/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model/HistogramSummary/valuesPackmodel/HistogramSummary/values/0model/HistogramSummary/values/1*'
_output_shapes
:�.*
T0*
N
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
P
model_1/pack/1Const*
dtype0*
value	B :.*
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
j
model_1/zerosFillmodel_1/packmodel_1/zeros/Const*
T0*'
_output_shapes
:���������.
R
model_1/pack_1/1Const*
dtype0*
value	B :.*
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
p
model_1/zeros_1Fillmodel_1/pack_1model_1/zeros_1/Const*
T0*'
_output_shapes
:���������.
R
model_1/pack_2/1Const*
dtype0*
value	B :.*
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
p
model_1/zeros_2Fillmodel_1/pack_2model_1/zeros_2/Const*
T0*'
_output_shapes
:���������.
R
model_1/pack_3/1Const*
dtype0*
value	B :.*
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
p
model_1/zeros_3Fillmodel_1/pack_3model_1/zeros_3/Const*
T0*'
_output_shapes
:���������.
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
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
~
<model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/splitSplit<model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split/split_dim0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add*
	num_split*
T0*@
_output_shapes.
,:	�.:	�.:	�.:	�.
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
_output_shapes
:	�.
�
4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�.
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel_1/zeros4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�.
�
1model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�.
�
3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�.
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�.
�
Emodel_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatConcatEmodel_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat/concat_dim2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2model_1/zeros_3*
_output_shapes
:	�\*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
~
<model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dimConst*
dtype0*
value	B :*
_output_shapes
: 
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/splitSplit<model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split/split_dim0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add*
	num_split*
T0*@
_output_shapes.
,:	�.:	�.:	�.:	�.
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
_output_shapes
:	�.
�
4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�.
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel_1/zeros_24model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�.
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�.
�
1model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�.
�
3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�.
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�.
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�.
[
model_1/concat/concat_dimConst*
dtype0*
value	B :*
_output_shapes
: 
x
model_1/concatIdentity2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*
T0*
_output_shapes
:	�.
f
model_1/Reshape/shapeConst*
dtype0*
valueB"����.   *
_output_shapes
:
k
model_1/ReshapeReshapemodel_1/concatmodel_1/Reshape/shape*
T0*
_output_shapes
:	�.
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
v
model_1/zeros_4Const*
dtype0*&
valueB�.*    *'
_output_shapes
:�.
�
model_1/VariableVariable*
dtype0*
shape:�.*
	container *
shared_name *'
_output_shapes
:�.
�
model_1/Variable/AssignAssignmodel_1/Variablemodel_1/zeros_4*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking(*
T0*'
_output_shapes
:�.
�
model_1/Variable/readIdentitymodel_1/Variable*#
_class
loc:@model_1/Variable*
T0*'
_output_shapes
:�.
�
model_1/Assign/value/0Pack2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model_1/Assign/value/1Pack2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�.*
T0*
N
�
model_1/Assign/valuePackmodel_1/Assign/value/0model_1/Assign/value/1*'
_output_shapes
:�.*
T0*
N
�
model_1/AssignAssignmodel_1/Variablemodel_1/Assign/value*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking( *
T0*'
_output_shapes
:�."	i�N��      x�{	�C{���A*�-

mean squared errorn�C=

	r-squared �:
�)
states*�)	   @z���   @�?    �&A!RSMG�@)����@2�S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]����ہkVl�p�w`f���n�w&���qa�d�V�_�6NK��2>�so쩾4>/�p`B>�`�}6D>ڿ�ɓ�i>=�.^ol>f^��`{>�����~>[#=�؏�>K���7�>���m!#�>�4[_>��>���]���>�5�L�>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�������:�              @      @      4@     �Q@     �i@     �@     ��@     0�@     d�@     <�@     D�@     ��@     ��@     �@     �@     ��@     ��@     ��@     /�@     W�@     R�@     �@     ��@     V�@     �@     ��@     ��@     ��@     �@     ��@     T�@     v�@     >�@     2�@     ��@     h�@     �@     `�@     \�@     3�@     ~�@     ��@     Է@     �@     ��@     ?�@     -�@     R�@      �@     ��@     h�@    �k�@    ���@    ���@     ��@     �@     �@     <�@     ��@     ��@     �@     �@     g�@     N�@     6�@     ��@     ��@     |�@     �@     ��@     �@     d�@     ��@     ��@     D�@     �@     ��@     ��@     �@     ȉ@     ��@     ��@     `�@     �@      @     �|@     �z@     @y@     0v@     Ps@     `q@     �n@     �m@     �h@     �j@     �c@     �b@     `a@     �^@     �\@     �\@     @X@     �U@      U@     �Q@      Q@     �L@     �M@      L@     �I@     �E@      B@      ;@      ?@     �A@      C@      6@      ;@      5@      0@      0@      &@      0@      .@       @      $@      &@      *@      @      "@      "@      @      �?       @      @      @      @      @      @      �?      @      @              @      @      �?       @      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?       @      �?               @      �?      @      �?       @      �?       @      @       @       @      @      @       @      @      @      @       @      "@      "@      @      @      .@       @      $@      5@      .@      ,@      1@      <@      9@      ;@      :@      B@      A@     �A@     �G@     �F@      O@      L@     �O@     �R@     @S@      T@     @U@      ]@     �_@     �[@      b@     �b@      e@     �c@     �e@     �j@     @m@     �p@     �q@      t@     �r@     �x@      y@     @~@     �}@     X�@     H�@     ��@     ��@     ��@     @�@     ��@     ��@     ȑ@     ��@      �@     ��@     \�@     ��@     ��@     ��@     ��@     ��@     ��@     D�@     6�@     ��@     ��@     M�@     ��@     ��@     �@     ߸@     �@     "�@    ���@    ��@    �d�@     N�@    ���@    ���@    �H�@     �@     ��@    ���@     �@     	�@     &�@     ̸@     ��@     ��@     p�@     2�@     ȫ@     ��@     ,�@     ,�@     �@     f�@     в@     !�@     ��@     V�@     ��@     ��@     ��@     �@     �@     5�@     )�@     ��@     ��@     #�@     ��@     d�@     ʸ@     ִ@     �@     N�@     �@     >�@     ��@     F�@     ��@     Е@     �@     �n@     �G@      @        
�
predictions*�	   @��R?   ��؋?     ί@! �.�B9@)�tx�C�?2�nK���LQ?�lDZrS?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�������:�              �?              @      @      @      3@      4@      D@     �F@      S@     �]@     `c@      i@     @h@      n@      s@     �u@     �{@     �}@     |@      y@     �q@      d@     �O@      0@      @        Ryآ       ��	ut*{���A*�A

mean squared error��@=

	r-squared�s�<
�,
states*�,	   ���   �~�@    �&A!��B��d�@)����@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R�����]������|�~���MZ��K���u��gr��R%������39W$:���
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
���H5�8�t�BvŐ�r�6��>?�J���8"uH���-��J�'j��p～
L�v�Q>H��'ϱS>��x��U>Fixі�W>w`f���n>ہkVl�p>f^��`{>�����~>T�L<�>��z!�?�>��ӤP��>�
�%W�>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>�u`P+d�>0�6�/n�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�               @      @      *@     �I@     �V@     �a@     �p@     pv@     ��@     Ȑ@     P�@     �@     ��@     ��@     ��@     �@     ��@     ]�@     ǲ@     D�@     |�@     ��@     n�@     �@     R�@     L�@     ��@     }�@     ��@     ��@     ��@     �@     л@     ��@     +�@    �J�@     ,�@     ��@    ���@    ���@    ���@     �@     e�@    �H�@    �&�@     ��@    ���@    ���@     �@     B�@     ��@     n�@     �@     9�@     Z�@     �@     �@     �@     ��@     ګ@     0�@     ��@     ĥ@     p�@     ̡@     8�@     Ԝ@     ��@     @�@      �@     ��@     8�@     l�@     ��@     ��@     X�@     H�@     ��@     @�@     �@     Ё@     �@      ~@     `z@      y@     @u@     �s@     �r@      o@      p@     �m@     `k@     `i@      d@      c@     �a@     @b@     �Y@      `@     �[@     @W@     @U@     @T@      Q@     �R@      Q@     �M@      I@     �H@     �B@     �A@      C@      A@      E@      9@      <@      =@      4@      7@      1@      6@      *@      3@      $@      .@      *@      "@      .@      @      "@       @      @      @      �?       @      &@      @      @       @       @       @      @      �?       @      �?       @              @      �?      @       @      �?      �?      �?      �?       @      @      �?              �?              �?      �?      �?              �?      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?      @               @      �?      �?       @      @      @              @              �?       @               @              �?      @              @      @      @       @      @       @      $@      @      @      "@      @      @      @      @      (@       @      (@      ,@      "@      ,@      &@      1@      ,@      2@      :@      5@      @@      5@      C@      ;@      ?@      B@     �I@     �I@      K@     @Q@     @Q@      R@      R@     �T@      T@     �T@     �[@      Z@     �^@     �c@     �a@     �c@      f@     �f@     `f@     `l@     �o@     �q@     0s@      w@     �w@     @x@     �z@     |@     @~@     @�@     ��@     ��@     ��@     `�@     ��@     Ȍ@     |�@     @�@     ��@     $�@     ��@     $�@     |�@     ��@     �@     �@     ��@     D�@     �@     ֩@     ��@     :�@     �@     �@     մ@     C�@     ��@     �@     ��@     �@     �@     �@    ���@     ��@    �8�@     ��@     ��@    ���@    ���@     ��@     m�@    ��@     J�@    � �@    ���@    �q�@    �"�@     n�@     ľ@     v�@     ;�@     >�@     ;�@     ɼ@     Ż@     �@     �@     �@     ��@     }�@     w�@     ��@     ��@     ü@     �@     �@     �@     �@     (�@     ��@     �@     ��@     ��@     ��@     ��@     �t@     @^@      3@      $@        
�
predictions*�	    >��   �^�?     ί@! ����>@)�O�@2�	�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[��a�Ϭ(���(���x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�	              �?      @      @      1@      F@     �M@     �P@      W@     @[@     @\@     �W@     �\@     @[@     @V@     �V@     �Q@     �P@     �P@      M@     �M@     �E@     �G@     �D@      E@      D@      :@      >@      6@      8@      1@      5@      .@      &@      $@      ,@      *@      .@      &@      "@      (@      @      @      &@      @      @      @      @      @       @      @      @      @      @      @      �?      �?      �?      �?      �?      �?      �?      �?              �?      �?               @      �?      �?      �?      �?      �?              �?              �?              �?              �?              @              �?               @              @       @      �?      �?      @      @      @      �?      @       @      @      @      �?      �?      @       @      @      @      $@      @      @      .@      @      @      @      3@      *@      (@      :@      =@      0@      1@      1@      <@      C@      :@      >@     �D@      I@     �F@      D@      L@      F@     �Q@     @S@     �P@     �Q@     �T@     �U@     �V@     �X@     �Z@     �[@     �[@     �_@     �^@     �Y@     �R@     �K@      E@      8@      @       @      �?      �?        ��ӟ�'      �&)	�W;{���A*�O

mean squared error�@=

	r-squared «<
�;
states*�;	   @=��    !5@    �&A!od3�Y�@)���\��@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J�/�p`B�p��Dp�@�����W_>�p
T~�;��so쩾4�6NK��2��'v�V,����<�)�4��evk'���o�kJ%��J��#���j�f;H�\Q������%�����X>ؽ��
"
ֽ5%����G�L���'1˅Jjw�:[D[Iu��-���q�        ����z5�=���:�=y�訥=��M�eӧ=�8�4L��=�EDPq�=����/�=K?�\���=�b1��=��؜��=i@4[��=z�����==��]���=��1���='j��p�=��-��J�=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>u 5�9>p
T~�;>����W_>>p��Dp�@>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>cR�k�e>:�AC)8g>ڿ�ɓ�i>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�               @               @       @      @      4@      I@     �]@     @p@     H�@     �@     
�@     �@     ��@     v�@     �@     <�@     G�@     Ю@     L�@     d�@     h�@     �@     ̭@     ��@     W�@     ��@     %�@     f�@     <�@      �@     ��@     =�@     Ϳ@     \�@    �Y�@     ��@     4�@     ��@    ���@    ���@     +�@     ��@     w�@     ��@     ��@    ���@    ���@     +�@     ؾ@     h�@     ��@     ��@     e�@     ��@     �@     ��@     $�@     ��@     ¨@     Ʀ@     �@     Ԣ@     
�@     R�@     ��@     �@     l�@     �@     \�@     8�@     @�@     �@     p�@     ��@     �@     x�@     h�@     x�@     0�@     h�@     X�@      �@     @~@     �}@     �y@     py@     �x@      w@     `u@     �r@     �t@     @q@     �q@     @m@      k@     �l@     �j@     �h@      i@     @g@     �c@     �c@     �`@     @`@     �]@     @Z@      ]@      V@     �R@     �W@     @U@      V@     �Q@     �P@      I@     �E@     @R@      I@     �K@      G@     �D@      >@     �A@     �@@     �@@      B@      B@      6@      =@      2@      5@      9@      (@      ,@      ,@      0@      3@      .@      .@      (@      "@       @      @      "@      $@       @      "@      &@      &@      @      @      *@       @      @      $@      @      @      @       @      @      @       @      @      @       @      @       @      �?              @      �?      @      @      @       @       @       @      @       @      @      �?       @      �?      �?       @               @              �?              �?              �?              �?       @      �?              �?       @              �?      �?       @              �?              @              �?               @              �?              �?              �?              �?              @              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?       @      �?       @              �?              �?               @      �?      �?       @       @       @       @       @      @       @      �?               @       @              @              @       @      �?      @       @      @       @      �?      @       @       @      @               @      @       @      @      @      @      @       @      @      $@      @       @      "@      @      @      @      $@      @      ,@      @       @      @      (@      1@      *@      .@      (@      7@      *@      3@      (@      *@      7@      5@      8@      8@      9@      =@      7@     �A@      B@      =@      ?@      ?@      =@      <@      J@      K@      B@     �B@      K@     �O@     @Q@     @Q@      T@     @Q@     �S@     �U@     @U@     �V@      Y@     �[@      \@     �V@      ^@     �`@     @c@      b@     �c@     �e@     �d@     `h@     @l@     �g@     �k@     �o@     `p@      r@     `s@     `s@     �q@     �s@     0y@     �z@     �|@     @}@     Ȁ@     H�@     Ё@     X�@     x�@     ��@     �@     h�@     P�@     �@     Ȏ@     ��@     �@     ��@     �@     ��@     ��@     ��@      �@     ̝@     ��@      �@     l�@      �@     ��@     ��@     ܪ@     J�@     ��@     B�@     �@     ��@     �@     ��@     9�@     '�@     �@      �@    ��@     ��@    �`�@     ��@     ��@     ��@     ��@    ���@     ;�@     ��@     ��@     �@     ��@    ���@    ���@    ��@     �@     ��@     =�@     �@     n�@     7�@     �@     ��@     �@     
�@     �@     ��@     ��@     M�@     p�@     V�@     ,�@     `�@     ��@     �@     z�@     8�@     �@     h�@     �u@     `h@     @U@      <@      ,@      @      �?      �?        
�
predictions*�	    5鬿   ��?     ί@!  �C}/�)��;W�$@2�	I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1�>�?�s���O�ʗ����uE���⾮��%���~]�[�>��>M|K�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�	              @      "@      4@      H@     �W@      a@     �e@     �d@      e@     �b@     `a@      ^@      `@     �Y@     �X@     �T@      L@      R@     �Q@     @R@     �K@     �N@     �G@     �B@      B@      9@      >@      .@      ?@      6@      .@      0@      8@      (@      &@      1@      @       @      "@      @      @       @       @      @       @      @      @      @      @       @      �?      @      @      �?               @       @      @      �?      �?      �?               @              �?      @              �?              �?               @              �?              �?      �?      �?              �?      �?              �?              @       @               @              �?       @      @      �?      @      @      @      @      @       @       @      @       @      @      @      @      "@      @       @      "@      (@      &@      (@      @      "@      "@      ,@      &@      ;@      2@      ,@      .@      8@      ;@      =@      <@      5@      C@      @@      A@     �E@     �A@      H@     �C@     �G@      I@      O@     �P@      Q@     @T@     �R@     @S@     @R@      Q@      Q@     �P@     �K@      A@      8@      &@      (@      $@      @       @       @        �7+8�-      `25	Y�J{���A*�Z

mean squared errorb/@=

	r-squared���<
�E
states*�E	    y1�    Ǯ@    �&A!\�2hӆ�@)�:C'�$�@2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO��������%���9�e�����-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[��PæҭUݽH����ڽ���X>ؽ��
"
ֽ�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6�����5%����G�L������6���Į#����8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�!���)_�����_����G-ֺ�І�̴�L�����-���q�        �-���q=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=_�H�}��=�>�i�E�=�8�4L��=�EDPq�=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�!p/�^�=��.4N�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�"              �?      @      �?       @      2@      K@     �U@      a@     @c@      g@     �y@     ȩ@     ȥ@     .�@     ʭ@     ,�@     �@     �@     @�@     >�@     z�@     ��@     �@     ��@     h�@     ��@     �@     0�@     ��@     ��@     ʻ@     Y�@     ſ@     ;�@     >�@     [�@     ��@    �5�@    ���@     ��@     j�@     ��@     ��@    �S�@     ��@     �@     ��@     ̼@     ݼ@     }�@     ��@     ��@     ��@     ��@     �@     3�@     `�@     |�@     �@     4�@     R�@     ��@     �@     �@     �@     n�@     ��@     ,�@     B�@     ��@     ,�@      �@     �@     �@     ��@     @�@     P�@     Џ@     ��@     Ȍ@     `�@     ��@     ��@     ��@     ؃@     p�@     h�@     ��@     ��@     p|@     0y@     �y@     �z@     �v@     �s@     �s@     `r@      q@     `q@     0p@     `o@     `k@     �o@     �j@      i@     �j@     �i@      f@     �e@      d@     `c@      d@     �a@     @]@     `a@     �b@      X@      [@      [@     @V@     �U@     �\@     @W@      Y@     �R@     �S@     �T@     �O@     �T@     �P@      L@     �K@      K@     �H@      J@      P@     �I@      G@     �K@      G@      K@     �E@     �A@      D@     �D@     �@@      C@     �@@      @@      >@     �E@      B@      ,@      9@      .@      ;@      8@      7@      1@      1@      ,@      0@      2@      0@      1@      .@      1@      3@      1@      $@      $@      *@      $@      &@      @      (@       @      $@      "@      @      @      &@      @      $@      @       @      @      @      @      @      (@      @      @      @      @       @      @       @      @      @      @      �?       @      �?      �?      @      @      "@       @      �?      @              @      @       @      @      @      @       @       @      @      @      @      �?               @               @      �?      �?      �?      �?              �?      �?      �?       @      �?               @              @      �?      �?              �?       @      �?              �?              �?              �?              �?      �?      �?               @      �?       @              �?               @      �?              �?               @              $@      @              �?              �?      �?              �?              �?              �?              �?      �?       @              �?              �?               @              �?              �?       @               @               @              �?      @      �?               @       @      �?       @      @               @              �?              �?       @      �?      �?      �?              �?      @       @       @      �?              @      @      @      @      @      @      @      @      @      @              @      @      @       @      @       @       @      @      @      @      @      @      �?      "@      @      @      @      @      &@       @      &@       @      "@      @       @      &@      "@      .@      "@      $@      @      .@      *@      1@      0@      (@      $@      ,@      &@      .@      1@      3@      0@      1@      8@      4@      6@      3@      ;@      ?@      <@      E@      <@      :@      D@      ?@      7@     �E@     �E@      F@      @@      G@     �M@      D@     �I@     �I@      H@     �J@      L@     �O@     �O@      L@      R@     �N@      N@     @R@     �R@     @T@     @S@     �T@     �T@     �T@     �W@     �Y@     �Z@      X@     �]@     �^@     @_@      ]@      _@      b@     @b@     �a@      d@      d@     �e@      g@     �g@      g@     �i@     �k@     �k@     `n@     0r@     �p@     0z@     �t@     px@     0w@     �v@     �x@     �z@     `{@     P|@     �}@     �@     @�@     ��@     (�@     H�@     ��@     8�@     Ї@     ��@     ��@     ��@     p�@     Ē@     <�@     <�@     �@     L�@     ��@     p�@     ��@     ��@     �@     ¡@     $�@     x�@     0�@     6�@      �@     �@     �@     ��@     Z�@     ر@     �@     �@     ٵ@     �@     ��@     ��@     ��@     �@     ټ@     ��@     �@     �@     �@     ��@     O�@    ���@    �u�@     ��@     ��@     ��@    �'�@     D�@    ���@    ���@    ���@     {�@     ��@     �@     ^�@     պ@     +�@     �@     �@     H�@     ߵ@     ��@     ��@     8�@     ��@     T�@     ��@     ��@     ��@     0�@     b�@     ��@     đ@     x�@     �w@     t@     �k@     @b@     @V@      5@      @      @      �?        
�
predictions*�	   �����   @��?     ί@!  ��u!U@)�t���@2�
��<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��>�?�s���O�ʗ����h���`�8K�ߝ�;�"�qʾ
�/eq
Ⱦ['�?��>K+�E���>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>>�?�s��>�FF�G ?1��a˲?6�]��?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?�������:�
              @      @      @      .@      2@      9@     �@@      D@      I@     �G@     �J@      C@     �C@      E@     �C@      @@      C@      D@     �A@      >@      7@      "@      1@      (@      3@      .@       @      $@      *@      &@      $@      "@       @       @      @      @       @      @       @      @      @      @      @       @       @       @      �?      @      @      �?       @      �?       @      �?              @      �?      �?              �?      @              �?      �?               @      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @               @              �?      �?               @      �?      �?       @               @      @              @      �?      @      @      @      @      @      @      "@      @      @      @      *@      &@      ,@      5@      "@      ,@      &@      *@      .@      6@      5@      4@      :@      =@      =@     �B@     �J@      B@     �@@      G@      O@     @P@      O@      V@     �U@     @V@      S@     @Y@     �\@      ^@     �`@     �d@     `c@     �e@      d@      a@     �a@     @a@      b@     �]@     �[@     �S@     �M@      @@      2@      $@      @      �?        �{�\�/      ]�[	t�[{���A*�_

mean squared error�>=

	r-squared ��<
�J
states*�J	   �Z� �   �H�@    �&A!P�8��
�@)\�6W��@2�$��tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q='1˅Jjw=x�_��y=%�f*=\��$�=�/k��ڂ=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�$              �?      @      @      .@      2@     �G@     @Y@     �p@     ��@     $�@     �@     :�@     �@     ��@     `�@     ʭ@     �@     ^�@     ��@     ǰ@     ��@     ��@     ��@     ɲ@     ��@     ��@     �@     ��@     /�@     ��@     �@     ٽ@     ��@     ǿ@     �@    ���@     ��@     Y�@    ���@     m�@    �&�@     ��@     ��@     R�@     8�@     ��@     ��@     �@     *�@     ��@     �@     ܴ@     R�@     �@     �@     6�@     ��@     ��@     J�@     ��@     �@     ��@     ^�@     ��@     H�@     T�@     Л@     ��@     ��@     ��@     ��@     @�@     �@     (�@     `�@     Ѝ@     ��@     �@     ��@     ��@     8�@     (�@     ��@     ��@     �@     8�@     X�@     ��@     �~@     ��@     |@     �y@     �x@     `w@     �u@     0u@     |@     @u@     Pr@     �q@     �p@     `p@     `l@      n@     �k@     @j@     �k@      k@     �i@     �h@     �g@     @d@     �e@      c@     @b@     @c@     �`@     �a@     �`@      `@     �d@     �^@     �[@     @]@     �\@     �Z@     �]@      X@      T@     �Y@      X@     @U@      X@     �R@     �Q@     @T@      P@     �V@     �N@     �P@     �J@     @P@     �Q@     @P@     @P@      O@     �K@     �K@     �Q@     �E@      H@     �F@     �D@      I@     �I@      >@      @@      I@      C@      >@      <@      @@      5@      9@      A@      @@      0@      ;@      4@      1@      5@      0@      1@      3@      3@      *@      3@      "@      6@      7@      &@      (@      "@       @      &@      (@      "@      &@      &@      "@      &@      (@      $@      @      "@      @      "@      $@       @      &@      $@      @      @      @      �?      @       @      @      @       @      �?      "@      @      @      @      @      @      @      @      @      @       @      @       @       @      @      @      �?      @      �?      �?      @      �?       @      @       @      @       @       @      �?      @      �?      �?      �?      �?      @       @       @      �?              �?      @      @      �?       @      �?      @               @               @       @      @       @              @       @      �?              �?      �?      �?      �?      �?      �?              �?              �?      �?       @              �?      �?      �?      �?              �?       @              �?              �?      4@      <@              �?               @      �?              @       @      �?              �?      �?      �?      �?      �?      @              �?              �?       @      @              �?       @      �?       @       @       @      �?              �?      �?               @       @      @      �?       @      @       @      �?       @      @       @      @      �?      �?      @       @      �?       @      �?       @               @       @      @      @       @      @              �?              @      @      @       @      @      @       @      @      @      @      @      @       @       @       @      �?      @       @      @      $@      @      @       @      @      @      "@      $@      "@      "@      $@      $@      "@      @      @      ,@      2@      @      (@      $@      1@      4@      2@      $@      1@      .@      ,@      1@      6@      7@      5@      1@      6@      4@      9@      7@      7@      =@     �@@      ;@      >@     �B@      @@      A@      <@      @@     �A@      F@     �D@     �F@      I@      E@      E@      E@      I@      G@     �J@      G@     �D@     �G@     �D@     �H@      P@      L@      O@      J@     @S@      R@     @R@     �R@     �Q@      T@     �Z@      Y@      U@     �T@     @V@      X@     �Y@      Y@      ^@     �Z@     �^@     �\@     �_@      ^@     �a@     �^@      c@     �a@      b@     �c@     �c@     �e@     �d@      f@      i@     @i@     �h@     �u@     �m@      l@     �n@     �m@     @n@     �o@     p@     �s@     �s@     �r@     pu@     �r@     �v@     �u@     �w@     �|@      {@      {@     `~@     �~@     ��@     ��@     @�@     �@     �@     ��@     ؅@     ��@     (�@     ��@     x�@     �@     x�@     \�@     ��@      �@     ��@     0�@     4�@     Ę@     l�@     $�@     ��@     `�@     �@     �@     �@     �@     �@     �@     p�@     �@     ��@     ��@     �@     �@     S�@     T�@     c�@     Q�@     u�@    �9�@     7�@    ���@      �@    �;�@     ��@     0�@    �V�@     ��@    ���@    �Z�@    ���@     ��@     w�@     ׾@     ��@     ��@     �@     �@     4�@     H�@     ��@     Ƕ@     е@     g�@     Q�@     r�@     ��@     ֲ@     ۲@     m�@     ��@     m�@     ��@     �@     +�@     ��@     6�@     ,�@     ��@      }@     �j@      V@     �I@     �M@      4@      @      @              �?        
�
predictions*�	   �(��   ���?     ί@!  ��@)S�&���@2�
�/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
������6�]���1��a˲�>�?�s���O�ʗ�����>M|K�>�_�T�l�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              (@     �L@     �]@     �^@     �d@     `b@     �Z@     �Y@     �W@      X@     �V@     @U@     �R@     @P@      V@     �T@      Q@     �R@     @Q@     �M@      K@      G@      G@     �D@     �E@      9@      A@      ?@      7@      *@      5@      1@      ,@      2@      1@      .@      $@      *@      "@      "@      @       @      &@      @      @      @      $@      �?      �?      @      @      @       @              @      �?      @       @       @               @      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?              �?      �?      �?      �?       @      @      �?      @      �?      @      @              �?      @      @      @      @      @      �?      @      @      @      "@      @      @      @      "@      *@      *@      .@      (@      .@      4@      (@      =@      6@      >@      A@      ;@     �@@     �C@     �A@      E@     �J@      D@      G@     �J@     �H@      G@     �H@     �I@     �C@      E@      F@      F@      M@     �I@      L@     �L@     �D@     �I@     �E@      H@      H@     �B@      <@      =@      6@      2@      4@      ,@       @      *@      "@      @      �?        �JN��/      A�G	`�n{���A*�_

mean squared error^3>=

	r-squared`g�<
�I
states*�I	   �����   �O�@    �&A!�S5_D�@)AF����@2�$ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ���@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�z����Ys��-���q�        �-���q=z����Ys=x�_��y=�8ŜU|=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�$               @      (@      *@      $@      F@     �U@     �q@     }�@     P�@     ��@     ��@     �@     ԰@     H�@     N�@     ְ@     ��@     ڲ@     ��@     ��@     P�@     �@     ��@     x�@     7�@     �@     �@     :�@     ׿@    �4�@     ��@     ��@     ��@    ��@    ���@    ���@     ��@    ���@    �'�@    �i�@    ���@    ��@    �<�@     �@     g�@      �@     
�@     ��@     ;�@     ��@     ϱ@     s�@     ��@     Ĭ@     �@     �@     r�@     v�@     ��@     ��@     ��@     `�@     `�@     �@     ��@     ܗ@     ��@     ��@     h�@     ��@     ��@     @�@     ȏ@     ��@     8�@     0�@     ��@     0�@     (�@     X�@     �@     p�@     ��@     @�@     �@     @~@     �z@     p|@     P}@      |@      x@     �v@      w@     u@     pt@     �v@     pt@     Pr@     �r@     �t@     s@     �r@     �y@     `q@     `n@      q@     �u@     `j@     �i@     �i@     @g@      i@      i@     @j@     `f@     �h@      e@     �d@     �b@     �c@     `c@     �b@     @b@     �a@      d@     �a@     �_@     �\@     �]@     �^@      ]@     �X@     �Y@     �[@     @^@     �Z@     @U@     @T@     @X@     @Z@     @[@      Y@     �[@     �V@     �S@     @R@      R@      S@     @R@     �R@     @P@     @S@     �P@     �O@     �N@     �L@      M@      J@     �P@     �N@     �K@      E@     �E@     �E@      D@      F@     �E@      D@      ;@      >@      A@      C@      B@      8@      5@     �B@      <@      9@     �A@      9@      ?@      >@      ,@      1@      3@      <@      5@      3@      *@      3@      &@      $@      &@      ,@      0@      *@      .@      3@      $@      ,@      .@      *@      "@      $@      (@       @      @      *@      .@      @       @      "@      @      *@      &@      @      @      @       @      "@      @      @      @      @      @       @      @      @      @      @      @      @      @       @      @      @       @              @      @      @      @      @      �?       @       @       @      @      @       @      �?       @              @      �?      @              �?              @       @      �?      �?              @      @      �?      @      �?       @              �?      @              �?              �?              �?      �?       @      �?               @              �?       @              �?      8@     �G@      �?              �?              �?              �?      @              @       @      �?       @      �?       @       @       @       @              �?      @      �?      �?               @      @      �?      @       @      @      @      @       @      @       @      �?               @      �?      @      �?      �?       @      @       @      �?              @       @      @      @      @              @      @       @      @      $@      @       @      @       @      @      @      @      �?      @      @      @       @       @      @      @      @      @      @      @      &@      "@      &@      @       @      "@      (@      "@       @      "@      @      &@      &@      "@      (@      4@      &@      ,@      (@      ,@      0@      3@      6@      3@      4@      7@      2@      5@      9@     �@@      1@      7@      9@      =@      >@     �B@      B@      >@      @@      C@      9@     �G@      B@     �A@     �D@      D@      C@      ?@     �G@     �L@      A@      I@      H@      E@      J@     �L@      M@     �L@      L@     �K@     �J@      J@      L@      Q@     �U@      J@      S@     �N@      Q@     �O@     �S@     @U@     �U@     �U@      Z@     @Y@     �U@     �W@     @[@     �T@      ]@      Z@     �]@     �`@     �`@      `@      a@     �_@     `b@     `a@     �a@     �b@      b@      a@      b@     @e@     �s@     �f@      g@      f@      i@     �i@     �j@     @i@     `n@     @n@     �m@      n@     �n@     �p@     p@     pp@     `q@     �r@     Pr@     �u@     �v@     �v@      x@     �w@     w@     �v@     `z@     P{@     �{@     p|@     �~@     h�@     ��@     X�@     ��@     ��@     �@     0�@     Ѓ@     P�@     H�@     X�@     @�@     �@     ��@     |�@     �@     `�@     T�@     ��@     Е@     �@     �@     h�@     �@     ��@     L�@     ��@     d�@     ��@     �@     ��@     J�@     ��@     "�@     �@     ��@     ��@     �@     a�@     ��@     �@     ��@     ؼ@     ~�@    ���@    ���@    ���@    ���@    ���@    �
�@    �y�@    �v�@     V�@    ���@     ��@    �[�@     ��@     ��@    ��@     >�@     ��@     �@     ��@     /�@     ��@     ��@     �@     �@     ��@     �@     �@     %�@     �@     �@     ԰@     ��@     _�@     ��@     ��@     ��@     ��@      }@      g@     @T@     �G@      B@      1@      "@      �?       @        
�
predictions*�	    EW��   `�_�?     ί@!  䅋u@)��!5�.@2�
� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����FF�G �>�?�s���})�l a�>pz�w�7�>>�?�s��>�FF�G ?����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?�������:�
               @     �P@     �e@     �f@     �i@     �d@      c@     @^@     �Y@     �Z@     �W@     @W@     �T@     �T@     �U@     �P@     �Q@     �M@      K@      H@      E@     �E@      E@      ?@      ?@      >@      5@      6@      6@      .@      &@      0@      $@      &@      &@       @      &@       @       @      @      @       @      @      @       @       @      @      @      @      @      @       @               @      �?              �?       @              �?      �?              �?              �?       @              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?      @      @      �?       @       @               @              �?      �?      @      �?      @      @      @      @      @      @      @      @      @       @      (@      "@      $@      "@      (@      *@      6@      2@      3@      7@      :@      1@      =@      5@      ?@      <@      >@      8@      <@      :@      <@      B@     �B@      ;@      >@      E@      8@     �B@     �D@      C@     �C@     �E@     �E@      C@     �G@     �H@      F@     �G@      E@     �K@      E@     �H@      A@      H@      ?@      ;@      7@      7@      2@      "@      (@      �?              �?        �rSY�0      '	��	��~{���A*�a

mean squared error�<;=

	r-squared�9=
�K
states*�K	   @�5��   �?@    �&A!�\��š@)�ɱ��@2�%ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�%               @      $@       @      .@      E@     @U@     p@     ��@     ֦@     ��@     ?�@     >�@     Ы@      �@     *�@     ��@     ~�@     ��@     ��@     Z�@     �@     ��@     N�@     X�@     �@     n�@     շ@     �@     ��@     I�@     j�@     	�@     |�@     E�@     ��@    �u�@    �@�@     �@    �s�@    ���@     �@     ��@     ��@     �@    ���@     ׿@     &�@     ��@     ��@     C�@     	�@     *�@     ѳ@     Բ@     Q�@     %�@     d�@     ��@     0�@     �@     ��@     (�@     �@     :�@     X�@     �@     T�@     `�@     x�@     ��@     ��@     ԕ@     ��@     ��@     ��@     �@     ,�@     ��@     ��@     ��@     �@     Ȇ@     ��@     ��@     ��@      �@     `�@     ��@     ��@     �@     �|@     P|@     p}@     �z@     �y@     �y@     �y@     x@      v@      u@     pt@     �s@     `t@     @r@     �s@      r@     pp@     0p@     �q@     0p@     pp@     �q@     0p@     `m@     @x@     �k@     @w@     �l@     �j@     �j@     �i@     �f@     �f@      h@     �e@     `h@     �i@     �d@     �a@     �d@     �c@     @c@     �d@      b@     @b@     �_@     �b@      `@     @`@     �^@     �^@     �\@     �Y@     �[@     @^@     @[@      Z@     �V@     @W@     �Y@     �\@     @\@     �W@     @W@      V@     �V@     @S@      W@     �S@     @S@      Q@     �W@     �S@     �O@     @U@     @P@     @P@     �P@     @P@     �O@      M@     �O@      N@     �L@      E@      N@      F@      M@     �H@      H@     �G@     �D@      D@     �A@     �C@      <@      @@      A@     �C@      7@      9@      ;@      <@      <@     �C@      >@      7@      ?@      4@      5@      2@      4@      ,@      2@      7@      4@      1@      0@      2@       @      (@      (@      3@      (@      0@      1@      .@      *@       @      @      "@      ,@      *@      @      $@       @       @      *@      "@      (@      &@      @      @      "@      @      @      @      ,@      @      $@      @      @      @       @      @      @      @      @      �?      @      @              @      @      @      @       @      �?      �?      @      @      @      @      @      @      @      @      @       @      @              @      �?       @       @      @       @              �?       @      �?      �?       @      @      �?              @              �?      �?      �?      �?       @      �?       @              �?      I@      J@              �?              @       @      @       @      �?      @              @               @       @      @      @      @      �?      �?      �?      @      �?      @       @      @      @       @       @       @       @              @              @              @      @      �?       @      @      @       @      �?       @              @      @      @      @      @      @      @      @      @      @      �?      @       @      @      @       @      @       @      @      @      @      @      @      (@      $@      0@      .@      &@      @      &@      *@       @      @      $@      @      ,@      *@      *@      2@      *@      2@      5@      0@      9@      1@      3@      4@      (@      4@      *@      6@      >@      1@      ;@      4@      >@      <@      <@      ;@      @@      @@      A@      ?@      9@      A@     �C@      C@      E@      <@      ?@     �D@      C@      @@     �A@      A@      E@      F@      E@     �K@     �K@      D@      D@      I@      P@      N@     �H@     �P@      P@      K@      H@     @R@     @Q@      M@     �R@      Q@      S@     �R@      N@      V@      P@     @V@     �S@      T@     @W@      [@     @V@     @Y@     �\@      \@     @\@     @`@      [@     ``@     @]@     ``@     �a@     �`@      _@      c@     �b@     @c@     �i@     �p@     �d@     `f@      a@      d@     �i@     �g@      h@      j@     `i@     �i@     �j@     `i@     �l@     @m@     @o@     �n@     �p@     �m@      r@     @r@     �p@     �q@     �t@     �m@     �t@     �s@     �s@     �w@     y@     �v@     �x@     0{@     @�@     �}@     z@     ��@     (�@     �@     ��@     @�@     ��@     @�@     ��@     ��@     �@     ��@     ��@     ؊@     �@     �@     ��@     `�@     �@     8�@     �@     �@     Ԗ@     8�@     Ș@     \�@     T�@     �@     Π@     *�@     �@     ܤ@     ��@     ��@     Z�@     �@     ��@     ��@     �@     %�@     ��@     �@     ��@     X�@     ��@     "�@     ν@     ��@     f�@     ��@     ��@     ��@    ���@    ��@    ���@    �g�@    ���@     (�@    �*�@     ��@     ��@     ��@     U�@     �@     �@     �@     :�@     -�@     1�@     ��@     ��@     �@     ϰ@     �@     B�@     ��@     ί@     ��@     Z�@     ��@     �@     ��@     ��@     ި@     ��@     �@     ��@      h@      T@      E@      5@      1@       @       @        
�
predictions*�	   �����   �b��?     ί@! ����"B�)K.�!��@2�
� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9��5�i}1���d�r�6�]���1��a˲�;�"�qʾ
�/eq
Ⱦ0�6�/n���u`P+d�����%�>�uE����>f�ʜ�7
?>h�'�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�������:�
              �?      �?      9@     �\@     @e@     �e@     �f@     @`@      ]@      ]@     �Y@     @\@     @Y@     �W@     �X@     �T@     �\@      ]@     @]@     �X@     @Y@     �S@      R@     �J@      N@      L@     �C@      C@      B@      A@      6@      7@      2@      5@      ,@      "@      ,@      .@      ,@      &@      @      "@      @      @      @      ,@       @      �?      @       @       @      @      �?      �?       @       @      @       @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?               @              �?              �?      �?              �?      �?      @      �?       @       @      @       @              @      @      @      @      @      @      @       @      "@      (@      @       @      $@      &@      (@      (@      "@      "@      1@      5@      7@      3@      :@      =@      ?@      ?@      ?@      =@      7@      C@      ;@      5@      @@      =@      6@      7@      >@      A@      =@      =@      =@      =@      ?@      8@      ?@     �@@      ?@      ?@      =@      6@      4@      .@      1@      &@      &@      (@      @      &@      @      $@      @      ,@      @      @      @        i�(�0      &�l	��{���A*�a

mean squared error�;;=

	r-squaredЯ9=
�L
states*�L	   � U�   @�`@    �&A! ���ů@)�|�r��@2�%�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              �?               @      @      2@      6@      >@      C@      S@      t@     �@     H�@     ֦@     f�@     (�@     ��@     4�@     6�@     ̫@     *�@     ��@     �@     ¬@     Z�@     �@     ��@     �@     ޲@     ?�@     ��@     ι@     �@     ��@      �@     Ͼ@     ��@    ���@    �G�@     ��@     m�@    �E�@    ��@     ��@     ��@     ��@     
�@     �@    �J�@     ؿ@     6�@     C�@     ��@     s�@     �@     ��@     '�@     \�@     ��@     ��@     ��@     �@     Ħ@     ��@     �@     �@     ֡@     ��@     ��@     l�@     |�@     d�@     T�@     �@     ,�@     ȗ@     X�@     ��@     ��@     Ē@     P�@     x�@     ��@     ��@     H�@     ��@     І@     Ȉ@      �@     ��@     ��@     ��@     h�@     ȁ@     h�@     8�@     0~@     �}@     `}@     �{@     y@     �y@     p{@     �{@      w@     Pw@     �u@     t@     �t@     �u@     s@     @q@      r@     �q@     `t@     �q@     0q@     �m@     �l@      o@     �o@     0r@     `l@     q@     �l@      k@     `i@      v@     �e@     �o@      t@     �f@     �c@      g@     `f@     @e@     �g@     @b@     �e@     @b@     �b@     �a@     �`@     @_@     �b@      `@     `b@     `b@     @_@     �`@     �`@     �\@     �]@     �Z@     �Z@     �[@     �]@     �]@     �^@      _@      Y@     �[@      Y@      X@      V@     �U@     @W@     �X@     �V@     @R@     @X@     �S@     �W@      Y@     �U@      S@      P@     �S@     �S@     �O@     �L@     �N@      O@     �M@     �Q@     �G@     @Q@     �F@      K@      F@     �H@      J@     �J@     �E@     �G@      I@     �C@     �B@      E@     �H@     �C@      A@      C@      <@      @@      9@      6@      <@      4@      5@      3@      >@      :@      7@      ?@      8@      7@      9@      6@      1@      1@      .@      (@      ,@      "@      4@      "@      "@      *@      ,@      (@      "@      ,@      &@      2@      "@      @      @      3@      &@      (@      $@      @      @      @      @      @      @      @      "@      @      "@       @      $@       @      @       @      @      @      @      @      @       @      @      @      @      @      @      �?       @      @              @      �?              �?       @      �?      @      @      �?       @      @       @      @       @      �?       @       @      �?      �?      @               @      �?       @      @       @       @       @      �?     �S@      L@      �?      �?      @              �?       @      @      �?       @              @       @              �?               @       @      @       @      �?       @      @               @       @       @      @       @      @       @      "@      �?              @       @      @      @      �?      @      �?      @      @      @      @      @      "@      @       @      @      @       @      @      @       @      "@      @      $@      $@      "@      &@      $@      *@      (@      @      &@      *@      &@      0@      @      &@      2@      ,@      $@      3@      5@      0@      3@      1@      3@      4@      1@      1@      5@      2@      1@      5@      4@      <@      5@      .@      1@      ;@      6@      ,@      @@      3@      9@      <@      D@     �@@      <@      G@      A@     �B@      ?@      4@      D@      A@      =@      A@      E@     �C@      C@     �M@      D@      G@      G@     �E@      M@     �J@     @Q@     �D@     �L@      H@     �K@      H@      L@      K@      N@     @S@     �T@     �U@     @U@     @R@      O@     �R@     �R@     @X@     �S@      R@     �X@     �U@     �U@     @X@     �V@      X@     �Y@     �X@     �^@      `@      `@     �Y@      Z@      _@      b@     �`@     �_@      q@      c@     �a@     �`@     �c@      d@     �e@     �d@     �d@     `f@      f@     `d@     �h@     @g@     `h@     `i@     `e@     �k@     �k@     `n@      o@      j@     �m@     `n@     @k@     @p@     �n@     �q@     �p@     p@      r@     0t@     �w@     u@     �t@      t@     Pt@     �v@     �x@     �|@     �x@      y@      {@     ��@     �y@     �y@     �}@     p{@     �|@     ��@     �@      �@     0�@     p�@     p�@     ��@     ��@     І@     �@     @�@      �@     Ȋ@     (�@     ��@     ��@     |�@     ��@     �@     0�@     d�@     �@     �@      �@     ��@     l�@     ��@     ��@     T�@     ~�@     L�@     ޤ@     ��@     J�@     ��@     «@     `�@     �@     �@     r�@     ��@     �@     N�@     ��@     �@     پ@     �@    ���@    ���@    �E�@    ���@    �j�@     ��@    ���@     ��@    ���@     ��@     ƿ@     x�@     %�@     �@     ��@     ;�@     j�@     �@     )�@     ��@     �@     ۲@     l�@     (�@     ��@     ��@     D�@     ��@     <�@     �@     h�@     �@     m�@     q�@     �@     �@     ��@     0�@     `l@     �Z@     @R@     �H@      6@      5@       @      @       @      �?        
�
predictions*�	   `黳�   ����?     ί@! �SUK�F@) �\���'@2�
� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��>h�'��f�ʜ�7
������6�]���x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?�������:�
              �?       @      @      $@      O@      \@      c@      e@     @a@      b@     �[@     @W@     �T@      Q@     �T@     �P@      O@      F@      E@      ?@     �B@      8@      9@      4@      ?@      :@      >@      6@      *@      5@      0@      2@      $@      @      (@      &@      @       @      @      @      @      @      @       @      @      @      @      @       @       @      @      @      �?      @      �?      �?              �?       @              �?              �?      �?              �?              �?               @              �?              �?              �?       @      �?              �?              �?       @      �?               @      @      �?       @      �?      @       @       @       @      @      @      @      @      @      @      @      @      @       @      @      $@      @      2@      (@      (@      @      (@      &@      (@      ,@      1@      3@      7@      9@      ;@      A@      ;@      ?@      B@      J@      B@     �J@      J@      N@     �T@     @S@     �T@     �V@     �U@     @V@     �R@      P@     �R@      O@     �P@      L@     �M@     �J@     �H@     �L@      G@      H@      C@      C@     �C@      B@      7@      >@      0@      (@      ,@      @      "@      @      @      @      �?        ��Y�1      !��	���{���A*�c

mean squared error��6=

	r-squared�5�=
�K
states*�K	    >��   ��@    �&A!ɣ|"y�@)�E09M�@2�%�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E�������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              �?              �?      @      $@      5@      ;@     �A@     @T@     �p@     �@     \�@      �@     ڭ@     �@     �@     @�@     ��@     ԧ@     H�@     �@     ֪@     ��@     <�@     �@     ^�@     h�@     �@     ǳ@     ��@     -�@     ��@     ;�@     R�@     �@     ��@     ��@     �@     E�@    ���@     �@    ���@     L�@    ��@    ��@    �u�@    �W�@     ��@     7�@     |�@     ˼@     a�@     ?�@     ��@     ��@     д@     X�@     O�@     W�@     Z�@     Ư@     ԭ@     J�@     ��@     ��@     Z�@     Ф@     ޣ@     L�@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     h�@     l�@     0�@     ��@     $�@     ��@     ��@     p�@     ��@     h�@     P�@     �@     Ȇ@     ��@     ��@     ��@     ��@     ��@     H�@     �@     @~@     P�@     @}@     �}@     `{@     pz@     @z@     `x@     �w@     �w@     �w@     �v@      u@     �t@     �r@     0r@     �r@      p@     �q@     `q@     Pp@     �p@      o@     �p@     �k@     �n@     �o@     @m@     �k@     �h@     �h@      j@      i@     �i@     �h@     �s@     �j@      i@     �q@     �e@     @d@      c@      b@     `b@     `b@     �d@     �d@      d@     �a@      `@      ]@      _@     @]@     �]@     �_@     �_@      ^@      _@     �`@     �`@     @^@     @[@     @]@      Y@      Z@      [@      Z@     �Z@     @U@      Z@     �V@     �^@     �V@      V@     �U@     �V@     �X@     �W@     �S@      S@     @W@      S@     �V@     �S@     �P@     �P@     �N@      S@      O@     �N@      L@     @S@     �Q@     �M@      F@     �L@      N@      G@     �H@      D@      O@     �B@      G@      H@     �B@     �B@      D@     �F@      B@      ;@     �E@      A@      >@      ;@     �B@      >@      4@      8@      <@     �A@      5@      ;@      ,@      5@      9@      4@      1@      2@      5@      .@      2@      *@      9@      &@      $@      (@      *@      *@      &@      ,@      &@      &@      &@      .@       @      @      &@      $@      "@      @      $@      @      &@      @      ,@      @      @      @      @      @      $@      @      @      @      @       @      @      "@       @      @      @      @      @      @      @      @      @       @       @      @       @      @              @      @      @      @      @      �?      @      �?      �?              �?      @      @               @      @              �?     �V@     �S@      �?       @      @      @      �?              @       @      @       @      �?      @              @      �?      @       @      @      �?      �?      @      @       @      �?       @      @      "@      @      @      @       @       @      @      @       @      @      @      @      @      "@      @      "@      @      @      @      (@       @      "@      @       @      @      *@      $@      ,@       @      (@      $@      (@      &@      2@      ,@      ,@      0@      *@      2@      4@      4@      1@      4@      0@      1@      .@      3@      3@      0@      0@      6@      7@      2@      5@      @@      =@      ;@      >@      >@      4@      =@      :@      9@      @@      @@      <@      >@      A@      >@     �A@      C@      A@     �C@     �H@     �D@      7@      C@     �E@      @@      E@      H@     �L@      H@      G@     �E@      K@      I@      M@      N@     �N@     �R@     �O@      Q@      R@     �G@      L@      N@      O@     �S@     �P@      U@     �T@     @U@     �S@     �R@     �W@      S@     @[@     @W@      W@     @R@      T@      W@      W@     @X@     @\@     �Z@     �Y@     �[@     �a@      `@     �b@     �o@     �a@     �a@     `c@      a@     @`@     �a@     �`@     �d@      b@      b@     �b@     `d@     `f@     �c@     �c@     @j@     `j@     @e@     �i@     �i@     @i@     �j@     @l@     �k@     `n@      m@     �j@     �n@     �p@     @q@      o@     0r@      r@     �r@     �q@     �q@     t@     �q@      u@     @u@     �y@     �y@     �u@     pw@     �w@     x@     `�@     �z@     �y@     �}@     �~@     �@     �~@     �~@      �@     ��@     ��@     ؃@     Ђ@      �@     ��@     ؆@      �@     ؋@     (�@     @�@     @�@     ��@     ��@     ��@     �@     ��@     �@     L�@     ��@     X�@     H�@     P�@     ��@     ��@     >�@     ��@     �@     @�@     6�@     �@     `�@     �@     
�@     f�@     �@     ��@     �@     ��@     ��@     m�@     D�@     �@     ��@     �@     ڽ@     �@    �'�@     ��@    ��@     ��@     6�@     &�@     ��@    ���@    ���@    �d�@     �@     ��@     ��@     �@     ,�@     7�@     ��@     �@     �@     c�@     #�@     >�@     ~�@      �@     n�@     ®@     :�@     r�@     ��@     �@     ��@     ��@     �@     p�@     s�@     �@     ��@     ��@     ��@     �p@     `a@      S@      N@      :@      6@      (@      @      @              �?        
�
predictions*�	   `�r��   ���?     ί@!  �z��)�P*��!@2��{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�;�"�qʾ
�/eq
Ⱦjqs&\��>��~]�[�>a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>O�ʗ��>>�?�s��>����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�������:�              �?      @      4@      M@     @`@     �i@     �g@     �f@      c@     @\@     �Y@     @X@     @W@     @\@     @Q@     �T@     �M@     �O@     �I@     �H@     �E@     �F@     �H@      B@     �B@     �F@      @@      B@     �A@      6@      :@      1@      6@      8@      ,@      2@      0@      *@       @      ,@      ,@      @      @      ,@      @      $@      @      $@      $@      @      �?      �?      �?      @              �?      @       @       @      @      @      �?              �?      �?      �?              �?               @              �?      @       @              �?              �?              �?              �?              �?              �?       @              �?              �?      @      �?       @      �?              �?      @      @      @      �?      @      �?              @      �?       @      @       @      @      @      "@      $@      @      &@      "@       @      .@      (@      (@      *@      1@      6@      *@      3@      5@      :@      C@      3@     �@@      8@      9@      B@      A@     �I@      A@     �B@      @@     �C@     �C@      F@      C@     �A@      F@      D@     �G@      E@      B@      A@      ?@     �C@      D@     �C@      @@      B@      =@      A@      7@      ;@      6@      2@      ,@      &@      1@      $@      ,@       @      @      @      @      @       @      @      @      �?      �?        �f��0      &�l	J��{���A	*�a

mean squared error-�5=

	r-squared�=
�L
states*�L	    ��   @ӯ@    �&A!&�4���@)���s��@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%               @               @      @      @      @      4@     �D@      N@     �Y@     0r@     �@     �@     T�@     ��@     f�@     �@     *�@     N�@     ��@     ̩@     ��@     ��@     &�@     ֬@     :�@     ��@     ޱ@     �@     ��@     M�@     �@     ��@     ��@     ҹ@     ��@     �@     ��@     ��@     ��@     ��@    ���@    �.�@     l�@    ���@    �c�@    ���@     ��@    ���@     �@     �@     ��@     ͼ@     c�@     �@     "�@     Ƶ@     ��@     Ų@     _�@     m�@     ��@     ��@     ��@     8�@     f�@     (�@     V�@     ��@     >�@     l�@     X�@     b�@     ��@     H�@     ��@     ��@     ��@     \�@     (�@     x�@      �@     <�@     @�@     ��@     �@     ��@     ��@     Ȉ@      �@     ��@     �@     ��@     (�@     Ȃ@     ��@     �@     p�@     ��@     �~@     �}@     �{@     @}@     �y@     @}@     �~@     Pv@     v@     �v@     0u@     0u@     pq@     pt@     s@     s@     0r@     �o@     �p@     p@     �p@      p@     `k@     �m@      n@     �m@      m@     �l@     �q@     �k@     `h@      h@     �e@      h@     �d@      g@     �g@     �e@     `g@     �p@      c@     �k@      i@     �c@     �a@      c@     `b@     �c@     �b@     �a@     ``@     @]@     �^@     �_@     @[@     �[@     @\@      [@      a@     �\@     �[@     �Z@     @]@     �X@     �^@     �Y@     �W@     @\@      X@     �V@     �Z@     �S@      T@      X@     @X@     �U@     @S@      Q@     @S@      X@     �V@      W@      Q@     �L@     �S@     @W@     �P@     �S@     �N@     @Q@      M@     �S@      J@      N@     �Q@     �Q@      J@     �E@     �K@     �J@      L@      M@      >@      F@      B@      A@      G@     �H@     �B@      C@     �D@      D@     �A@      B@      7@     �C@      =@      8@      <@      <@      ;@      ?@      9@      &@      A@      A@      8@      :@      2@      :@      ,@      6@      8@      =@      (@      *@      3@      4@      .@      $@      "@      *@      .@      @      2@      .@      3@      (@      ,@      (@       @      @      @      @      "@      "@      &@      $@      &@      &@       @      "@      @       @      "@      @      @       @      @      �?      @      @      @      @      @      @      @      @       @      @      @      @      @      @      @      �?      @      @      �?       @       @       @              �?      �?       @      �?      @       @              @             �`@      V@      @      �?       @      @              @      @      @      @      @      @      �?       @      @      �?      @      @      @      @               @      @      @      @      "@      @      @      $@      @      @      @      @       @      @      &@       @      @      @      &@      @      (@      @      &@      $@      "@      ,@      "@      2@      @      &@      ,@      "@      @      ,@      6@      ,@      .@      $@      ,@      6@      6@      1@      .@      8@      5@      0@      (@      5@      2@      :@      4@      3@      ;@      4@      3@      2@      :@      1@      9@      :@      2@     �@@      ?@      >@     �@@     �B@      A@      =@      7@     �D@     �@@      F@      >@      8@     �G@      B@     �G@     �G@     �D@      O@      B@      J@      F@      I@      I@      L@     �I@     �J@     �I@      M@     �F@      O@      M@      N@     �O@      N@      P@      N@     �K@      Q@     @R@     �R@     �T@      R@      R@     @R@     @T@     �U@     �S@      X@     �T@     �U@     @X@     �W@     @[@     @W@      W@     �T@     �\@     �\@     �Z@     @]@     �m@     �^@     �\@     �[@     �^@     @^@     `a@     �^@     �`@     �`@     �^@     �a@     �c@      c@     `b@     �b@     �b@     �g@      c@     �f@     �d@     �j@      h@     �j@      h@     �j@     @h@      j@     �g@      n@      m@     �j@     �i@     �p@     `n@     @p@     `k@     �n@     �n@     pq@     @r@     pq@     �r@     �r@     @u@     �y@     �t@     �t@      v@     t@     �w@     �v@     X�@     �z@     Pz@     @{@      {@      |@     ؀@     �@     h�@     ��@     p�@     �@     x�@     ��@     ؃@     ؃@     ��@     X�@     ��@     (�@     ��@     `�@     H�@     ،@     @�@     h�@     �@     ��@     ��@     ��@     T�@     p�@     ��@     �@     ԛ@     ��@     ��@     v�@     ��@     ��@     إ@     ڥ@     ħ@     
�@     �@     ��@     �@     +�@     �@     !�@     L�@     ��@     <�@     ^�@     �@     ּ@     �@     ��@     ��@    ���@     ��@    ���@     ��@    �%�@    �5�@    ���@     N�@     m�@     ��@     ��@     F�@     �@     �@     ��@     j�@     �@     �@     ��@     G�@     �@     ��@     w�@     b�@      �@     �@     b�@     ��@     ��@     �@     �@     ��@     ��@     �@     j�@     ��@     
�@      z@      e@     @Y@     @P@     �G@      6@      .@       @       @      �?        
�
predictions*�	    ���   ��B�?     ί@! ���Ƃ;@)��Q�[1@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�6�]��?����?f�ʜ�7
?>h�'�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�
              �?       @      3@      G@     �V@     �b@     �d@     �d@     �^@     �`@     �_@     �Z@      X@     �X@      R@     �P@     �R@     �Q@     �R@     �P@      N@      E@      F@      E@      D@      E@      C@     �B@     �@@      5@      7@      4@      6@      0@      (@      .@      .@      $@       @      "@      @      @       @      @      @      @      @       @      @       @      �?      @      @      �?       @              �?      �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?       @      @       @       @      �?      �?      �?      @      �?      @       @      @      @      �?      @      @      @      @      @      @       @      @      @       @      &@      ,@      @      2@      (@      (@      3@      (@      0@      0@      6@      7@      1@      :@     �C@     �A@      <@      J@      F@      H@      D@      G@     �O@     �H@      H@     �C@      D@     �H@     �H@     �C@      H@      J@     �B@     �@@      K@      H@      D@     �B@      E@     �B@     �@@      @@      <@      @@      6@      5@      &@      7@      "@      &@      @      @      @      �?      @      @      �?      �?        ��N91      ~���	���{���A
*�a

mean squared error3�3=

	r-squared ��=
�L
states*�L	    �l�   �4�@    �&A!�RXJ1�@)�ǐE��@2�%�Š)U	�h�5����tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              �?               @      ,@      0@      1@     �@@      K@     �[@     v@     ��@     ��@      �@     j�@     6�@     N�@     \�@     B�@     ��@     ��@     Χ@     V�@     8�@     ��@     ��@     ��@     ȫ@     ̯@     �@     ��@     m�@     ��@     ��@     ��@     >�@     <�@     ��@     W�@     &�@     ��@     ο@    �l�@    ���@     R�@    �#�@    ��@     ��@     \�@    ��@     ^�@     D�@     �@     �@     k�@     ��@     �@     p�@     ��@     Y�@     ٱ@     T�@     ��@     V�@     B�@     <�@     0�@     ��@     �@     ��@     ܡ@     ��@     H�@     �@     �@     �@     ��@     �@     �@     D�@     Ȕ@     Ԓ@     �@     �@     x�@     ��@     p�@     ȋ@      �@     `�@     ��@     ��@     ��@     0�@     ؂@     p�@     ��@     0�@     x�@     ��@     �@     �}@     �}@     �x@      {@      x@     �x@     �}@     �x@     �w@     �r@     u@     t@     �r@     �q@     q@     Pp@     `o@     @o@     �m@     �o@      m@      m@     �p@     �p@     �h@     �h@      i@     �g@     �j@     �f@     �h@     @h@      g@     �e@      f@     �c@     �f@      e@     �b@     �p@     �d@      a@     �a@      k@      n@     `a@     �]@     �a@     �`@     �`@     �[@     �[@     �[@      ^@     @]@     �^@      X@     @Y@      `@     �[@     �[@      Z@     @S@     �W@      W@     @Z@     @[@      Y@     @U@     @V@     �U@     @V@     @V@     �W@     @S@     @W@     @U@     �X@     @R@      S@     �R@     @P@      O@     �S@     �R@     �R@     @P@     �J@     �O@     �L@      L@     �K@      Q@     �P@     @P@      L@      N@      L@     �N@      E@     �I@     �H@      I@      G@     �E@     �C@      B@      D@     �C@      G@      D@     �C@     �@@      B@      =@     �D@     �@@      <@     �A@     �B@      ?@      =@      4@      >@      :@      ;@      7@      6@      5@      9@      8@      3@      ;@     �A@      5@      6@      2@      3@      (@      5@      $@      0@      .@      *@      .@      ,@      1@      *@      &@      *@      &@      @       @      (@      $@      $@       @      @      $@      @      "@      "@       @      *@      @      (@      &@      @      @      @      @      @      @      @      @       @      @      @      @      @      @       @      @      @      @      @      @      @       @      @      @       @       @      @      @       @      @      @              �?     �c@     �c@      @      @              @      @      @      @      @      @       @      @      @      @      @      @      @      @      @      $@      @      @      $@      $@      @      $@      @      @       @      @      $@      @      @      @      $@      @       @      "@      @      .@      $@      (@      *@      ,@      1@      (@      9@      2@      4@      6@      .@      3@      .@      7@      8@      2@      9@      2@      5@      1@      4@      0@      1@      1@      :@      ;@      =@      5@      9@      >@      :@      9@      7@      :@      >@      1@      B@      8@      >@      ?@      C@     �@@     �E@      D@      B@     �E@     �C@     �D@      C@      C@     �C@     �D@      D@      F@      D@     �G@      D@      J@      F@      E@     �N@      K@     �J@      H@      N@      I@      Q@     �O@      L@      K@     @P@      O@     �Q@     �G@      L@     �T@      S@     @Q@     �P@     �Q@     @S@     �S@     @W@      S@     �P@     @Q@      W@     �P@      V@     @V@     @W@     �V@     �V@     �Y@     �X@     @Y@     @m@      `@      ^@     �[@      Y@      ^@     @Y@     �\@     �\@     �Y@     @`@      `@     �`@     �`@     @^@     ``@     �d@     `b@     �a@     `d@     �d@     @g@     �b@     @f@     �d@      d@     `c@     �h@      g@      k@     �k@      i@     `i@     �i@     �j@      l@      l@     �k@      m@     pp@     �m@     �o@     �p@      q@      q@     �r@      s@      v@     �v@     �r@     pr@     �u@     �t@     �v@      z@      {@     {@     @|@     p}@      y@     �x@     �x@      {@     �~@     (�@     P@     p�@     �@     �@      �@     x�@     p�@     �@     ��@     `�@     ��@     ��@      �@     0�@     (�@     h�@     �@     4�@     x�@     ��@     �@     ��@     x�@     ��@     ,�@     ��@     �@     t�@     p�@     ~�@     ��@     ��@     ��@     b�@     �@     ��@     ��@     �@     �@     y�@     N�@     ��@     V�@     ,�@     ��@     [�@     չ@     ǻ@     ��@     �@     C�@     N�@    ���@    ���@    ��@     ��@     ��@    ���@    �D�@     _�@     7�@     ��@     "�@     m�@     <�@     ҹ@     �@     }�@     i�@     ַ@     ,�@     Ӱ@     ��@     L�@     ��@     ��@     ��@     b�@     x�@     �@     �@     �@     ��@     \�@     t�@     �@     :�@     ��@     ί@     P~@      h@     �\@      O@      A@      ;@      2@      @      �?      @      @      �?        
�
predictions*�	   �Ƿ�   ����?     ί@! �)$=FI�)�q˭L%@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�������.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�
              @      2@     �J@     �R@     �]@     �j@     �j@      l@     �h@     @d@     �`@      `@      a@      \@      \@     �X@     @[@     �R@     �S@     @P@     @R@     �N@     �L@      I@     �D@     �B@     �A@     �C@      6@     �C@      @@      6@      5@      2@      1@      &@      2@      &@      .@      .@      @       @       @      @      @       @      @      @      @       @      @      @      @       @      @      @      @       @       @              �?      �?       @              �?      �?              �?      �?      �?      �?              �?      �?              �?      �?      �?               @              �?              �?       @       @               @      @       @              @      @      @      �?      @      �?       @      @      @      @      @      @      @       @      @      .@      *@      "@      (@      "@      @      (@      (@      (@      8@      8@      0@      7@      1@      7@      8@      8@      4@      7@      1@      :@      7@      :@      7@      <@     �@@      ;@      :@      9@      5@      9@      ;@     �@@      >@      <@      :@      3@      2@      5@       @      "@      2@      ,@      @      "@      *@      &@      @      @       @      @      @       @      @       @               @               @        s5�EB1      ��N	6��{���A*�b

mean squared errorÎ0=

	r-squared Q�=
�K
states*�K	   ����    et@    �&A!3q;�皳@){�i��~�@2�%w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%              �?      @      $@      4@      8@     �O@     �Z@     �f@     py@     }�@     8�@     6�@     $�@      �@     �@     ��@     �@     ��@     D�@     ��@     ��@     h�@     ��@     ��@     ��@     ��@     ��@     ű@     .�@     �@     ٸ@     n�@     �@     ��@     ��@     �@     c�@     �@     7�@     �@     ��@    ���@    �'�@     ��@    ���@    �*�@     ��@    �u�@     �@     r�@     ��@     �@     �@     V�@     е@     v�@     w�@     �@     ��@     ڮ@     B�@     ��@     �@     0�@     x�@     @�@     *�@     R�@     ��@     b�@     4�@     ��@     ��@     �@     �@     ��@     �@     @�@     ؕ@     ��@     H�@     `�@     H�@     H�@     ��@     h�@     ��@     ��@      �@     ��@     ��@     0�@     ��@     �@     ؂@     x�@     ��@     x�@     X�@     P@     }@     �{@     �z@     �y@     @y@     0w@     px@     �w@     �v@     �z@     Py@     �u@     �s@     �t@     `s@     Pr@     r@     0q@     �p@     �m@     `p@     �n@      m@     �m@     �k@     @m@     `m@     `h@     �i@     �j@     @k@      h@     @j@     �m@     �h@     �h@     @h@     �f@     �f@     �d@     �s@     @c@     �c@     @_@     �a@     �a@     �d@     �o@     �c@     @]@      \@     �_@      b@     @\@     @a@     @`@      ^@     �]@     �U@     �Z@      \@      ^@     @\@     @Y@     �X@      V@     �[@     �R@     �W@     �Y@      U@     @W@     �W@     �Z@     @X@      Q@     @T@     �W@     @V@     �O@     �S@     �V@     �P@      S@     @T@     �Q@      T@      M@     @S@     �P@      P@      R@     �M@     �L@      P@     @Q@      Q@     �J@      F@      K@     �H@      C@      M@     @P@     �I@     �C@     �C@      G@      F@     �D@      E@      E@      E@     �D@      F@      B@      >@      C@      C@      C@      D@      =@      A@      8@      @@      4@      :@      @@      A@      2@      :@      >@      9@      4@      4@     �A@      6@      0@      2@      =@      9@      3@      .@      5@      $@      7@      2@      0@      .@      1@       @      *@      2@      *@      *@       @      &@      2@       @      0@      ,@      .@      $@      "@       @      "@      &@      "@      (@      @      @      @      "@      *@      @      "@      @      @      @       @      &@      @      @      @      @       @       @      @      @      @      $@      @      @      @      @      @      @       @      @      f@     �g@      @       @      @       @      @      @      @      @      @      @      @      *@      @       @      @      @      @      @      "@       @      *@      "@      &@      @      @      @      @      $@       @      (@      .@      .@      .@      3@      4@      (@      1@      3@      (@      *@      0@      .@      *@      5@      *@      ,@      ,@      6@      5@      ,@      6@      8@      3@      3@      6@      9@      8@      1@      6@      2@      ;@      5@      ;@      ,@      7@      ;@     �@@      7@      <@      =@      :@      >@      <@      9@     �D@      <@      ?@      ?@     �A@      =@      >@      >@      E@     �D@     �A@     �A@      E@      D@      H@      H@      J@     �C@     �D@      I@      E@     �E@     �G@     �I@     �L@      E@     �I@     �A@      L@      J@     �G@      Q@     �K@     �Q@     �Q@     @Q@      K@     �Q@     �Q@      M@     �L@      M@      M@     @P@     @S@     @U@     @P@     �N@      U@     �U@     @W@     @S@     �T@     @S@     �Y@     �U@      ^@     �i@     �\@      W@      \@     @X@     �W@     @[@     �`@     �]@     �]@     @\@      `@      `@     �a@     �a@      ^@     �^@     �a@     �_@     �d@     �a@     `b@     �`@     `e@     �`@     @f@     �d@     �f@     �g@     �d@     �g@     @f@     @g@     �e@     �f@     �j@      m@     �k@     �k@     �m@     `i@      i@     �m@     �o@     @o@     Pq@     �q@     �p@     �p@     �p@     �t@     x�@      y@      u@     �t@      w@     �u@     �v@     �x@     �w@     `x@     Py@     �y@     �z@     �z@     p{@     �~@     8�@     �@     �~@     �|@     ��@      �@     ȁ@     ��@     8�@      �@     ؈@     �@     ��@     8�@     X�@     ��@     ؊@     (�@     @�@      �@     ��@     \�@     l�@     ��@     �@     ��@     �@     d�@     T�@     `�@     l�@     ��@     ��@     V�@     ��@     z�@     ��@     B�@     ��@     z�@     �@     0�@     �@     8�@     L�@     ִ@     ��@     �@     =�@     :�@     �@     ��@     ;�@     �@     ��@    �I�@    ��@     .�@    ���@     ��@     w�@     Y�@     ��@     ��@     Ⱥ@     `�@     �@     i�@     ��@     )�@     \�@     �@     �@     �@     ��@     ¯@     <�@     X�@     ά@     �@     ��@     6�@     ή@     ��@     ®@     ��@     !�@     &�@     ��@     6�@     &�@     B�@     �y@     �c@     @^@      R@      I@      6@      5@      @               @        
�
predictions*�	    ����   ���?     ί@!  l�L�5@)8ڑ��2@2�
��(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x�������6�]����5�i}1?�T7��?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�
              �?      �?      @      5@      ;@     �B@     �R@     `a@     @b@      h@     @g@     �f@     �b@      `@      \@     @U@      T@     �P@     �R@     �L@      N@     �E@      I@      ?@      @@     �E@      ?@     �@@      ,@      5@      6@      9@      8@      4@      6@      4@      ,@      .@      (@      "@      @      @       @      @      @       @      "@      @               @      @       @      @      @              @      @      �?       @       @       @       @      �?      �?      �?              �?               @      �?      �?      �?               @              �?      �?              �?              �?              �?      �?      �?      �?               @       @      �?       @      �?      �?              �?       @       @      @      @      @              �?       @      @      @      @      @      @      @      @      @      @      *@      2@      @      ,@      ,@       @      *@      $@      ;@      *@      ,@     �C@      ;@     �B@      ?@      @@     �A@      ?@      D@     �C@     �J@      H@     �B@      D@      G@      D@     �@@     �B@     �D@      F@      F@      D@      D@     �E@      B@      D@     �E@     �A@      ;@      C@     �@@     �H@      D@      @@      ;@      9@      7@      2@      @      "@       @      @      @      &@      @       @      @       @       @        ��AY�1      |�4�	��{���A*�c

mean squared error�D+=

	r-squaredP�>
�L
states*�L	    X�   `4 @    �&A!���	[�@)�rY3�<�@2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%              �?       @      �?      @      @      "@      7@      5@      M@     �Z@     �i@     �z@     �@     z�@     ܥ@     ��@     r�@     ��@     �@     T�@     j�@     Ī@     ��@     ��@     Ь@     X�@     R�@     X�@     ��@     >�@     d�@     ��@     �@     	�@     �@     ع@     �@     ɹ@     ��@     :�@     �@     e�@     `�@     �@     :�@    ���@     �@     �@    ���@    �e�@     ��@    ���@     ��@     _�@     ��@     ��@     y�@     ��@     ��@     r�@     Ʋ@     d�@     ��@     t�@     @�@     �@     F�@     ��@     $�@     ��@     8�@     ʠ@     J�@     ��@     ��@     ̙@     ��@     (�@     �@     ��@     ��@      �@     ��@     ��@     D�@     L�@     ��@     �@     X�@      �@     Ȋ@     ��@     Њ@     ��@     ��@     ��@      �@     �@     ��@     ��@     Ȁ@     ��@     �@     0}@     0~@     �@     �}@     y@     Py@     pz@     y@     �x@     �v@     �x@     `y@     @t@     @v@     0t@     @r@     �s@      r@     �s@     �r@     �o@     Pr@     0q@     �m@      m@     �n@     �l@     �k@     @j@     �l@      i@     �k@     �h@     �i@     `g@     �i@     �g@      j@      f@     �e@     `a@     ps@     `g@      i@     �j@     @a@     �b@     �`@     `c@     `l@     `d@     �c@      a@     @`@      ^@     @a@     @\@      `@      Y@     �^@     �]@     �W@      `@     �_@      Z@     �Z@      ^@      [@     �X@     @W@     �Z@     �W@      X@     �V@     �R@     @X@      S@      T@     @X@     �T@     �S@     �T@     �T@      T@     �U@     �U@     �W@      Q@      R@     �N@     �Q@     �R@     �P@     @S@     �P@     �R@     @R@     �M@      P@      I@     �N@     �P@     �J@      L@     �C@     �G@      J@      P@      M@     �B@     �H@      H@      J@      G@      G@     �@@     �G@      A@      D@      J@      ?@     �E@      C@      >@      ;@      =@      :@      <@      :@      9@      :@      =@      8@      3@      <@      ,@      8@      3@      5@      5@      6@      <@      4@      4@      1@      8@      :@      3@      8@      0@      2@      ;@      1@      0@      0@      .@      $@      .@      .@      @      $@      0@      $@       @      "@      *@      &@      ,@      @      $@       @      (@       @       @      @      @      .@      @      "@      @       @      @      @      @      @       @      @      ,@      @      @      �?      @      �?       @       @      @       @      @      @     `l@     @j@       @       @      @      @      @       @      &@       @      @      (@      @       @      $@      @      @       @      $@      @      @      0@       @      "@      ,@      @      "@      &@      "@      (@      $@      (@      $@      $@      "@      ,@      ,@      1@      0@      (@      2@      .@      1@      .@      (@      4@      ,@      3@      4@      .@      1@      0@      4@      ,@      4@      4@      6@      0@      <@      <@      0@      7@      4@      5@      9@      8@      .@      9@      4@      9@      A@      A@      1@      :@      8@      ?@      8@      C@      ;@      @@     �@@     �K@      >@     �B@      @@      F@      A@     �@@      H@     �I@      G@     �D@      G@      E@     �A@     �E@     �J@      E@      N@     �E@      K@     �G@     �M@      B@      B@      M@      H@      H@     �O@      Q@     �I@     �G@     �F@     @Q@      M@      K@     �O@     �Q@     �R@      R@      O@     �H@     �U@      U@     @T@     @W@     @U@     @W@      U@     �V@     `e@     `b@     @X@      W@     @X@     �W@     �Y@      [@     @Y@      X@     �_@     @\@     �\@     �\@     �[@     @\@     @Z@     @a@     �a@     �^@      b@      `@     �b@     @a@     �f@      b@     �c@     �b@     �c@     �d@     �c@     �d@      d@     �f@      f@     �c@     �h@      f@     �j@      k@     �i@      l@     `l@     �m@     �m@      l@     �o@     �p@     �m@     @r@     �s@     �y@     �v@     s@     @r@     s@     �r@      s@      v@     0s@     �t@     @w@     @w@     �v@     �x@     �z@     �x@     Px@     `|@     �@      {@     �}@     X�@     (�@     �@     �~@     ��@     ��@     ��@     Ȇ@     8�@     H�@     ��@      �@     p�@     (�@     �@     P�@     ��@     ؍@     ��@     �@     $�@     T�@     t�@     ܕ@     ��@     ��@     ,�@     `�@     0�@      �@     ��@     L�@     ��@     ��@     V�@     J�@     l�@     h�@     t�@     x�@     9�@     ��@     �@     ~�@     a�@     (�@     "�@     �@     ߽@     Q�@    ���@    �X�@    ��@     ;�@    �1�@    ���@     ��@     ��@     Z�@     ��@     �@     �@     �@     ��@     M�@     !�@     ��@     ȷ@     @�@     Ҷ@     ��@     	�@     ��@     ��@     ԭ@     f�@     ��@     ��@     ��@     H�@     ��@     :�@     v�@     h�@     �@     �@     ۴@     ܬ@     ު@     �@     ��@     �n@     �^@     �R@      @@      =@      .@      @      @      @        
�
predictions*�	   @7���   ��_�?     ί@! �Q7�>�)���M.�*@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���f�ʜ�7
��������(��澢f����a�Ϭ(�>8K�ߝ�>6�]��?����?x?�x�?��d�r?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?�������:�              �?              @      @      0@      8@      <@      F@     �O@      S@     �Y@     @^@     �a@      d@     �`@     �e@     `c@     �d@     �f@      `@     �]@      ^@      [@     @Y@     @T@      V@     �O@     @Q@     �E@      H@     �M@      G@     �D@     �@@      6@      8@      1@      2@      ,@      $@      &@      "@      .@      .@      @      $@      @      &@      @      @      "@      "@      @       @      @      @      �?      @      @      �?      @              �?       @      �?      �?              �?      �?              �?              �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?               @      �?              �?       @               @      @       @       @      @      @      @      @      @      @      @      $@      @      @      @      @      @      "@      @      .@       @       @      $@      &@      2@      (@      3@      7@      4@      5@      3@      8@      5@      *@      6@      1@     �B@      9@      D@      D@      8@      <@      =@      5@      :@      2@      ;@      4@      8@      6@      ;@      :@      5@      <@      8@      0@      8@      2@      .@      .@      0@      2@      @      *@       @      @      @      @      @      @      @      @      �?       @              �?              �?        B� ?�2      ����	���{���A*�e

mean squared error >#=

	r-squared��+>
�L
states*�L	   `��   ��:@    �&A!U[��A�@)*��W���@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              @      �?      @      @       @      2@      6@     �I@     �X@     �h@     p{@     "�@     ��@     ��@     ��@     ة@     �@     L�@     ��@     ��@     2�@     ��@     �@     v�@     �@     ��@     ��@     Z�@     �@     :�@     m�@     I�@     +�@     ��@     D�@     4�@     ܺ@     	�@     [�@     ޺@     ��@     x�@     9�@     ߾@     ��@    ���@     :�@     ��@     ��@    ���@    �q�@    ���@     y�@     ��@     ׺@     �@     �@     ��@     ��@     ۲@     $�@     )�@     Ȭ@     �@     l�@     �@     h�@     b�@     ��@     ޢ@     ��@     |�@     8�@     `�@     ��@     ��@     ��@     �@     ��@     0�@     �@     ��@     ��@     T�@     D�@     (�@     p�@     �@     Ȋ@      �@     @�@     �@     P�@     ��@     ��@     �@     �@     ��@     `�@     `�@     x�@     `�@     �{@     �|@     @~@      |@     �}@     0z@      |@      y@     �t@     �w@     �v@     �y@     �{@     0t@     �u@     Ps@     0s@     pq@     �p@     p@     �p@     �q@     �q@     0p@     �q@      m@      k@     �m@     �g@      h@     �g@     �h@      i@      j@     �l@     �f@     �o@     �r@     �f@     �e@     �c@     �a@     �d@     �b@     �b@     `d@     �k@     `c@     �a@     �a@     �c@     `l@      f@      b@     �a@     @]@     ``@     �`@     �\@      _@      ]@     @\@     @]@     @[@     �Y@     @]@     @W@     �^@     @X@     �Z@     �T@      X@      X@      W@     �T@      V@     �W@     �U@     @Y@     �X@     �S@      T@     @W@     @T@      R@     �Q@     �T@      W@      P@     �P@      M@     �R@     �S@     �Q@     @P@     �O@     �O@     @R@      M@      L@      R@      I@      Q@      N@     �H@     �K@     �E@     @R@      H@      K@      ?@      H@      E@      D@     �C@     �C@      B@      C@     �G@      A@      C@      A@      >@     �A@     �B@      <@      =@     �A@      C@      :@      6@      @@      5@      A@      9@      7@      <@      .@      ?@      <@      4@      5@      0@      9@      0@      3@      4@      ,@      :@      6@      3@      9@      .@      $@      .@      0@      .@      (@      .@      (@      2@      2@      &@      "@       @      0@      &@      *@      1@      *@      @      ,@      @      (@      $@      @      @      &@      @      (@      @      *@      (@       @      $@      @      @      @      @      @       @      @      @      �?       @      @       @      @       @     �m@      n@      "@      @      1@       @       @       @      $@      @      @      @      "@      @      @      *@      "@      ,@      @      $@      @      "@      *@      $@      ,@      "@      @      @      .@      0@      &@      $@      (@      5@      ,@      0@      (@      *@      6@      .@      .@      "@      (@      (@      6@      3@      3@      ,@      (@      .@      3@      5@      7@      8@      5@      5@      7@      9@      7@      2@      3@      4@      8@      <@      @@      9@     �@@      9@      5@      ;@      8@      =@      9@      =@      ?@     �A@     �B@      9@      C@      ;@     �D@     �@@      >@      E@      B@     �C@     �A@     �C@      D@      H@      A@      F@     �A@      <@     �E@     �E@     �B@     �C@      G@     �J@     �C@      K@     �G@      J@      K@     �L@      K@      K@     �N@      H@      L@     �H@     @Q@     �J@      Q@      Q@     �K@     �R@     �O@     �Q@     �R@      O@      S@      S@     �P@      Q@     �T@     @U@     �T@      T@      e@     �`@     �X@     �U@     @V@     �U@     �W@     @\@     �[@     �Y@     @W@     �[@     �[@     �X@      [@     �^@     @\@      \@      `@     �]@     �]@     �a@     @b@     �a@     �b@     �a@      b@     @b@      e@     �d@     �d@     �b@     @e@     �g@     �d@      f@     @d@     �g@     `g@      i@     �h@      i@     �l@     �g@     �p@     0u@     �m@     �k@      m@     �p@     Pv@     �q@     r@     �q@     @r@     �p@     r@     s@     0r@     u@     �u@     u@      u@     �s@     �w@     �w@     �z@     pw@     pw@     ��@     �}@     }@      @     �@     ��@     ��@      �@     @�@     ��@     h�@     x�@     8�@     `�@      �@     ��@     ��@     ��@     �@     ��@     x�@     Ȏ@     ��@     ��@     ��@     ��@     ��@     ��@     �@     (�@     �@     �@     �@     �@     �@     �@     x�@     p�@     �@     ̪@     "�@     ��@     ��@     (�@     3�@     �@     6�@     6�@     ĸ@     Ժ@     (�@     ��@     M�@     ��@     ��@    ��@     ��@     ��@     �@     h�@     �@     ��@     ��@     ��@     ��@     /�@     �@     �@     F�@     "�@     g�@     /�@     T�@     |�@     ��@     X�@     �@     ��@     |�@     ��@     |�@     �@     4�@     ��@     :�@     1�@     ��@     �@     d�@     r�@     \�@     �@     "�@     X�@     @n@     @_@     �P@      E@      =@      3@      *@       @      �?      @        
�
predictions*�	   ��Ŀ   @���?     ί@!  D:B@)�f�"�6@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
������1��a˲���[���FF�G �>�?�s���I��P=��pz�w�7��jqs&\��>��~]�[�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>1��a˲?6�]��?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?�P�1���?3?��|�?�������:�              �?      (@      "@      :@      9@      >@      E@      K@     �G@      M@      Q@     @S@     �S@      U@     @Y@     �Z@      V@      X@      X@      [@     �Y@     @[@     �W@     @V@     �P@      S@      N@     �M@     �D@      F@      B@      =@     �D@      ?@      =@      5@      9@      5@      >@      5@      2@      3@      ,@      *@       @       @      @      @      *@      (@      @      @      @      @      @       @      @      @      @      @      @       @       @      @       @               @      @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @       @              �?              �?              �?       @       @              @      �?      @       @      �?      @      @      @      �?      @      @      @      @      @      @      *@       @      "@      (@      $@      "@      "@       @      $@       @      1@      2@      2@      2@      =@      8@      @@     �@@      ;@     �G@      A@     �D@      I@     �B@      F@     @P@      P@     �G@      P@      A@     �L@     �K@      A@      I@      ;@      F@      B@      >@      4@      E@     �A@      7@      9@      6@      6@      :@      3@      4@      7@      8@      0@      .@      "@       @      (@      ,@      @      @      @      @      @       @      �?      �?      @      @               @              �?        ͪ�2      ��!s	B�|���A*�c

mean squared erroreN=

	r-squared�sE>
�L
states*�L	    ;��    G@    �&A!��r���@)��e��@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%               @              @      @      @      *@      1@     �E@      R@     �a@      v@      �@     ��@     �@     �@     f�@     |�@     ��@     �@     �@     ��@     ��@     t�@     ��@      �@     ��@     ��@     P�@     Z�@     ��@     ��@     p�@     ?�@     `�@     a�@     �@     P�@     h�@     g�@     '�@     o�@     )�@     ��@     R�@     ��@     ��@     ��@    ���@    ���@     $�@    ���@     �@    �6�@     M�@     {�@     S�@     ��@     Ǵ@     -�@     S�@     ��@     ��@     8�@     �@     ��@     �@     �@     ��@     ��@     ��@     ��@     z�@     ��@     ��@     �@     ��@     ��@     �@     Д@     ̔@     x�@     <�@     ȑ@     (�@     �@     ��@     ��@     Ȍ@     ؊@     �@     ��@     X�@     `�@     ��@     h�@      �@     0�@     h�@     ��@     �@     8�@      �@     ��@     �@      ~@     �z@     Pz@      x@     `z@      x@     p{@     p}@     �u@     �u@     @w@     Pt@     `r@     �r@     `t@     Pr@     �q@     pr@     0p@     �n@      p@     0p@     �o@     `o@     @l@     @l@     �o@     �k@     �k@     �g@     �n@     �r@     @g@     `h@     `j@     �k@     �e@     �c@     �e@     `e@     `d@     �c@     �f@      a@     �b@     �b@     �_@     �b@     `q@     �c@     `e@     `d@      g@     �a@     �\@      c@     @`@     �]@     �[@     �\@     @_@     �Y@      Y@      [@     �X@      X@     @[@     @Z@      Z@     @T@      V@     �T@     @V@     @X@     �[@     �V@      U@     �Q@     �T@     �S@     �S@     @T@     @R@      W@     �T@     �N@     �T@     �R@     �T@      N@     �O@     @P@      N@     @S@      M@      P@     @P@      L@     �G@     @T@     �K@     �I@     �K@      N@      M@     �C@     �@@     �M@     �J@      E@      I@      K@     �E@      C@     �G@     �E@      D@      7@      B@     �E@      D@      3@      >@      ?@      ;@      >@      5@      B@      :@      B@      5@      :@      9@      5@      @@      <@      5@      6@      :@      <@      .@      8@      =@      ,@      2@      &@      3@      &@      2@      ;@      .@      3@      0@      2@      1@      "@      $@      3@      1@      *@      0@      &@      .@      (@      (@      &@      (@      "@      .@      &@      @      @      .@      *@      $@       @      &@      @      @      &@      @      $@       @      @      @      @      @      @      "@      @      @      @              @      @      @      @      o@     `q@      @       @      "@      @      (@      "@      @       @      (@      @      @      @      (@      @      $@       @      "@       @      @      (@      @      $@      (@      1@      ,@      $@      $@      2@      @      2@      "@      ,@      0@      .@      &@      5@      .@      .@      0@      0@      3@      1@      6@      2@      $@      1@      (@      7@      6@      .@      5@      3@      4@      6@      9@      5@      0@      8@      <@      9@      >@      2@      3@      =@      <@      2@      B@      ?@      2@      :@      <@      ?@      >@     �@@      ;@      A@      @@      A@      A@     �C@     �B@     �A@     �E@      <@      >@     �C@      H@     �B@      D@      =@      =@      A@     �K@     �J@      E@      L@      L@     �B@     �F@      G@     �I@     �H@     �F@      J@     @Q@     �G@     @Q@     �Q@     �M@     �O@     �G@     �Q@     @Q@      N@      P@      P@      S@     @R@     �M@     �R@     @R@     @T@     @S@     @Q@     �X@      U@      Z@     �S@     �V@     @X@      T@     �X@     @V@     �V@     �Z@     @X@     �U@     @X@     �U@      h@     �g@     �[@     �[@     @a@     �a@     �^@      ]@     @_@     �[@     �^@      d@     �c@     @c@     �b@     �_@      b@     `f@     �c@     �f@     �c@     �b@     �d@     �c@     �g@     �e@     @e@     �h@      f@     �i@     �s@      q@     �k@     �l@      o@     �o@     �k@     @u@     �o@     q@     �p@     �q@     �p@     �p@     �q@     �p@     �q@     �t@     �s@     �s@      w@     @u@     �x@     �w@     `z@     �w@     �y@      }@     ��@     �~@     p}@     P@      @     �~@     �@     `�@     P�@     0�@     (�@     Є@     �@     Ȇ@     �@      �@     ��@     8�@     �@     ��@     Џ@     �@     P�@     T�@      �@     t�@     �@     D�@     ��@     ؝@     X�@     ��@     ~�@     ��@     ��@     �@     ��@     ��@     ��@     ܪ@     ��@     ��@     0�@     ��@     U�@     O�@     ��@     ��@     U�@     r�@     �@     ��@    ���@     ��@     ��@    �R�@    ���@    ��@     h�@     C�@     Ľ@     ü@     y�@     �@     |�@     {�@     }�@     ��@     �@     1�@     ��@     4�@     i�@     #�@     ǰ@     P�@     �@     ^�@     &�@     �@     ��@     *�@     ή@      �@     ,�@     �@     ��@     �@     ��@     d�@     T�@     ��@     �@     ��@     �s@     �h@      [@     �H@      G@      7@      *@      @      @      �?        
�
predictions*�	   @6Fο   ��l�?     ί@!  �y���?)�0�G|�?@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?yL�����?S�Fi��?�������:�              �?               @       @      @      @       @      2@      @@      C@     �F@     �E@     �J@     �Q@      V@     @Z@     @\@     �`@     �a@     �]@      a@      _@     �`@     �]@      ]@     @`@     @V@     �V@     @T@      I@     �K@     @P@     �E@      F@      @@      8@      9@      :@      ;@      1@      7@      (@      (@      0@      .@      "@      @      "@      "@      @      @      *@      @      @      @      @      @      @      @       @       @       @      @      �?      �?              �?      �?      �?               @              �?      @      �?              �?      �?              �?              �?               @      �?              �?              �?              �?       @      @              @      @      @      @               @       @      �?      @      �?       @      @      @      @      @      @      @      @      @      @      @      @      &@      &@      6@      $@      &@      0@      4@      <@      8@     �C@      =@      9@     �@@      C@      A@      D@     �B@     �G@      K@      F@     �B@     �G@      B@     �F@      E@      I@      ;@      =@      3@      @@     �@@      ?@      ?@      <@      ;@      >@      7@      .@      ,@      1@      (@      .@      3@      2@      *@      ,@      @      &@       @       @       @      @       @       @      @      �?      �?      �?       @      @      �?      �?              �?        Z*F�22      �H�	�b|���A*�d

mean squared error"=

	r-squared`�1>
�L
states*�L	   ���   �~�@    �&A!�)��$��@)N\���@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              @      @       @      @      @      1@      @@      H@     �T@     �d@      x@     ޭ@     P�@     ��@     ��@     J�@     ��@     �@     ��@     ��@     ��@     <�@     j�@     ��@     <�@     ^�@     ��@     D�@     ��@     ,�@     2�@     �@     Y�@     �@     7�@     ��@     �@     <�@     F�@     ��@      �@     ��@     ��@     ��@     j�@     i�@     ��@    �@�@    ���@    �	�@     ��@     �@    �g�@    ���@     R�@     ��@     {�@     %�@     ��@     Y�@     ��@     ��@     ��@     �@     Ȩ@     f�@     ��@     �@     ʣ@     P�@     ��@     ��@     ��@     ��@     (�@     ��@      �@     ��@     ,�@     ��@     4�@     Б@     ��@     ��@     ,�@     ��@     ��@     `�@     ��@     �@     H�@      �@     @�@     ��@     Є@     �@     H�@     ��@     (�@      �@     ��@     ��@     ��@     �@     (�@     p|@     �{@     {@     �w@     Pz@      y@     �w@     �u@     pw@     �u@     �u@      s@     �r@     �q@     �q@     @q@     0q@     �p@     0p@     0p@     �p@      l@      n@      n@     �p@     �i@      k@      i@     �j@     `h@      j@     `h@     �g@     �g@     �f@      d@     @g@     `f@     `t@     �c@     `c@     �b@     �e@     �a@     `a@     �q@     �a@     @a@      `@      b@     �a@     `a@     �^@     �^@      `@     �^@     �_@     @]@     �[@     �X@     �^@     �]@      [@      ^@     `e@     �Z@     @[@     @X@     @[@     �Z@     �X@     �R@      V@     �S@     �Y@     �R@     �X@     @U@     �P@     @T@     �W@      Q@      R@     �Q@     �R@      Q@     �S@      U@     @R@     �L@      Q@      T@      M@      P@     �K@     �M@     �Q@      M@      M@     �K@     �L@     �K@     �F@     �J@     �K@      F@      O@      D@     �I@      K@     �D@     �K@     �G@     �G@     �H@     �D@     �G@      K@     �G@     �C@      B@      :@      G@      =@      <@      9@      <@      =@      <@      ?@      =@      5@      @@      =@      ;@      =@      7@      =@      0@      3@      @@      7@      .@      7@      4@      :@      9@      ?@      2@      6@      (@      0@      4@      0@      .@      4@      *@      9@      (@      &@      "@      $@      @      $@      1@      1@      ,@       @      $@      $@      .@      $@       @       @      "@      @      &@      &@      @      @      (@      @      @      @      @       @      @      @      @       @      @      @       @      @       @      @     �p@     �p@       @      @       @      &@      @      &@      @      @      ,@      @      "@      &@      @      "@       @      @      @      &@      "@      @      ,@      3@      &@      @      *@      $@      *@      .@      @      1@      "@      $@      "@      .@      (@      2@      2@      (@      ,@      .@      2@      0@      3@      2@      3@      7@      5@      :@      1@      2@      4@      2@      .@      5@      3@      2@      .@      ;@      7@      5@      5@      <@      7@      3@      B@      8@      ;@      7@      9@     �C@      9@      2@      8@      8@     �D@      =@      ?@     �B@      F@      >@     �D@      8@     �D@     �G@      D@     �D@      G@     �D@      D@      E@      G@      G@     �L@     �D@      J@      D@      I@     �G@      G@     �G@      I@     �G@      K@     �N@      I@     �N@     �O@     �K@      O@      N@     @Q@     @S@     �L@      P@      O@     @S@      V@     @R@      T@      R@     �P@      Y@     �R@     �S@     �V@     �R@      W@     @S@     �W@     �Q@      [@     �W@     �X@     �W@     �Y@     �]@     �W@     �]@     @_@     @V@      [@     @\@     @Z@     �p@     �^@      c@      \@      `@     �`@     �`@     �]@     �`@      b@     �b@     �b@     �d@     �e@     @e@     �d@     �b@      d@     @e@     �c@     �d@     `f@     �s@     `g@      h@     �k@     �k@     �i@     �g@      k@     �i@     �m@     �t@     �o@     �m@     pp@     �q@     s@     @q@     �r@     �r@     ps@      t@     �q@     �s@     �u@     @t@     �v@     Pw@     @v@     @v@     �{@     `y@     py@     �~@     �|@     ��@     ��@     0�@     ~@     �@     ��@     ��@     ��@     �@     @�@     H�@     ��@     0�@     �@     ��@     Ћ@     ��@     D�@     ��@     ��@     �@     ��@      �@      �@     �@     X�@     4�@     ԝ@     �@     j�@     8�@     ��@     >�@     ȣ@     "�@     *�@     ��@     �@     �@     ��@     Z�@     *�@     ��@     :�@     ʷ@     �@     W�@     0�@     ��@    ���@     ��@     J�@    ��@    �x�@    ���@     ��@     �@     U�@     ��@     ��@     2�@     Z�@     (�@     /�@     �@     ��@     ��@     +�@     ��@     Ȳ@     ��@     ��@     ��@     ��@     ��@     4�@     Ʃ@     ��@     ��@     $�@     �@     ��@     v�@     ��@     Z�@      �@     ��@     #�@     �@     ��@     ;�@      �@     Pw@     `k@      a@     �R@     �I@      4@      ,@      "@      @      @        
�
predictions*�	   `&�ȿ   @G�?     ί@! ���z\V�)�a��6@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&���ڋ��vV�R9��T7����5�i}1���d�r�f�ʜ�7
�������_�T�l׾��>M|Kվx?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?yL�����?S�Fi��?�������:�               @       @      .@      7@      C@      D@      E@     �H@     @R@      W@      \@      `@     `a@     @`@     `c@     �d@      f@      c@     �c@     `b@     @d@     �a@     �`@     �Y@      ]@      T@     �L@     �P@     �O@      J@     �I@      H@      ;@      <@     �D@      0@      :@      5@      8@      (@      ,@      .@      4@      "@      $@      @      @      "@      �?      @      �?      @      @      @      @      @      �?       @       @               @      �?      �?      �?       @      �?       @       @      @              �?               @              �?      �?              �?              �?              �?               @              �?              �?               @      @              �?      �?              @      @      �?      �?      �?      �?              @      �?      �?      @      @       @      @      @       @      @       @      @      @      @      $@      $@      (@      $@       @      @      &@      2@      (@      ,@      0@      ,@      3@       @      6@      5@      6@      8@      0@      5@      3@      .@      4@      3@      3@      1@      :@      2@      8@      ,@      3@      3@      (@      (@      ,@      4@      .@      "@      &@      .@      "@      $@       @      @      @      @      @       @      @      @      @      @      @               @      �?      @      �?       @       @              @      �?              �?              �?        ܢ.�1      |�4�	*r+|���A*�c

mean squared error`i=

	r-squared��D>
�L
states*�L	   �C��   ���@    �&A!�|���@)�hSz��@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%               @              @      �?       @      *@      4@      C@      Y@     �f@     0x@     >�@     >�@     ��@     ®@     h�@     n�@     ��@     ��@     ȩ@     ,�@     ʧ@     ��@     �@     p�@     �@     ��@     j�@     ��@     :�@     (�@     ��@     �@     c�@     X�@     �@     �@     ��@     �@     1�@     ��@     ��@     I�@     �@     k�@     Ľ@     )�@     ��@     ��@     ��@    ��@     ��@    �a�@    �(�@     1�@     ��@     ;�@     ��@     <�@     �@     O�@     �@     d�@     ��@     ��@     0�@     Z�@     �@     >�@     ^�@     �@     ��@     \�@     ؜@     4�@     �@     |�@     ��@     ��@     ,�@     ��@     ��@      �@     <�@     8�@     ��@     ،@     @�@     `�@     ��@     ��@     X�@     �@     ��@      �@     8�@     ��@     0�@     �@     ��@     ��@     �@     �@      �@     �@     }@     `z@     pz@     P~@     �x@     �x@      w@     @v@      x@     �t@     �s@     u@     �t@     @w@      u@     @r@     �r@     �q@     0p@      p@     @r@     Pq@     `p@      m@     �m@      j@      k@      k@     �k@     �i@     `g@     �g@     @f@     �h@     `e@      e@      k@     @h@     �r@     �f@     `b@     `f@     `d@     `b@     `b@     �a@      b@     @]@     @b@     �\@     @\@      j@     �f@      _@      `@     �[@      _@     �\@     �Z@     �`@     �\@     �V@     �[@     �_@     @[@     �Z@      Z@      X@     @\@      [@     �X@     �Z@     `e@      S@     �V@      U@     @U@     �S@     @X@     �R@     �T@     �Q@     @S@     @Q@     @S@      T@     @R@     �Q@      R@      G@     �Q@      W@      J@     @Q@      Q@      G@      G@      Q@     �M@      R@      P@     �O@      O@      M@      Q@      O@     @P@     �D@      F@      I@     �N@      D@     �J@      G@      ?@      E@      E@     �C@      I@      G@      G@      9@     �D@     �A@     �H@      D@     �H@      F@      ?@     �C@     �D@      B@     �A@      C@      C@      =@      =@     �@@      :@      <@      9@      9@     �@@      7@      5@      6@      3@      8@      9@      8@      7@      .@      2@      0@      6@      ,@      4@      4@      "@      @      $@      @      &@      .@      1@      0@      2@      &@      @      (@      @      $@      ,@      (@      0@      @      ,@      @       @      (@      @      *@      "@      @      (@       @      &@       @      @      "@       @      ,@      ,@      @      $@      @     q@     �q@      &@      &@      @      @       @      @              @      @      @      @      &@      @       @      "@      $@      $@      &@      $@      ,@      "@      @      $@      @      $@      3@      "@      ,@      2@      $@      "@      (@      *@      &@      (@      &@      (@      .@      0@      ,@      (@      5@      (@      .@      *@      .@      1@      $@      7@      *@      7@      3@      0@      >@      6@      .@      6@      3@      9@      3@      1@      6@      <@      7@      1@      >@      4@      >@      6@      6@      8@      :@      >@      @@     �A@      @@      >@      A@      ?@      A@      @@     �A@     �@@     �C@      F@     �E@      @@     �E@     �G@     �B@     �@@      E@      I@      D@      D@      F@     �G@     �J@      L@      K@      F@      Q@     �J@     �O@     �K@      S@     @P@     �K@      Q@     �K@      J@     �Q@     �P@     �R@     �V@      P@     @S@      S@     �R@     �R@     �K@      T@     �P@     �W@     @S@     �U@      U@     �V@      U@      V@     @V@      X@      Y@     �[@      Z@      Y@     @Y@     �[@     �[@     �]@     �Z@      \@     �[@     `b@      _@     �\@     @[@     �m@     �d@     �a@     �^@     �`@      a@     �a@     �b@     @c@     @c@      d@     �d@     �e@     �e@     �i@      p@     �e@     �d@      i@     @h@      j@     �j@      k@     �g@      k@      k@     @l@     @v@     @p@     �m@     r@     p@     �o@     �o@     �q@     �r@     �r@      r@     `q@     0s@     �t@      u@     �w@     �v@     �w@     `x@     �v@     �w@     �{@      z@     �z@     ��@     `~@     �@     �@     0�@     ؀@     ��@     (�@     ��@     Ѕ@     h�@     �@     ��@     ��@      �@     �@      �@     X�@     ؎@     ��@     �@     Ԓ@     �@     p�@     d�@     p�@     0�@     ��@     ��@     ��@     0�@     D�@     T�@     ��@     `�@     b�@     Z�@     �@     :�@     ��@     P�@     ��@     �@     ��@     ��@     �@     ׹@     ͻ@     �@     �@     п@    �n�@     �@     }�@     ��@     ��@     :�@     D�@     ��@     ռ@     ��@     �@     ��@     ]�@     ȼ@     .�@     û@     o�@     �@     ��@     ��@     ��@     e�@     Ю@     V�@     N�@     �@     H�@     �@     $�@     ��@     <�@     ��@     ��@     ��@     [�@     �@     ��@     4�@     &�@     J�@      �@     p�@     �t@     �i@      b@     �U@      N@      >@      3@      �?      �?       @        
�
predictions*�	   ��ѿ   ��@     ί@!  �9��Z@)�֖��L@2�_&A�o�ҿ���ѿ�Z�_��ο�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��h���`�8K�ߝ�1��a˲?6�]��?����?>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��bȬ�0?��82?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?w`<f@�6v��@�������:�              �?      �?              @      @      @      @      $@      &@      (@      2@      =@     �C@      ?@     �J@      P@      P@     �W@     �X@     �[@     �]@     @U@      Y@      X@     �Y@     �X@     �T@      N@     �P@     �Q@     �O@     �M@     �D@      C@      >@      ?@      >@      9@      ,@      5@      *@      ,@      0@      *@      $@      @      @      &@      @      @       @       @      @      @      @      @      @      �?      @      @      @      @               @      �?       @      �?       @       @               @      �?      �?      �?       @              �?              �?      �?              �?              �?       @               @              �?              �?              �?       @      @      @      @      @       @      @      �?      @      @      @      @      @      @      �?      @      @      @       @      4@      @      .@      (@      "@      *@      0@      4@      9@      8@      3@      7@      8@      ?@      B@      @@     �D@      E@      I@     �F@     �N@     �Q@      M@      P@     @S@     �P@     �I@      I@      I@      M@     �P@     �M@      G@     �G@     @P@      I@      I@     �D@      D@      C@      G@      ?@      :@      9@      5@      A@      ,@      6@      &@      *@      (@      .@      ,@      @      @      @      @               @      @       @       @      �?      �?              �?        �"��1       �[�	��;|���A*�c

mean squared error@�=

	r-squaredc>
�L
states*�L	   ����   `e@    �&A!Ε����@)�誒N�@2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%              �?      �?               @       @      *@      6@     �@@     �I@     �S@     �d@      ~@     ѱ@     &�@     �@     ܮ@     �@     (�@     z�@     2�@     >�@     \�@     :�@     ��@     ��@     R�@     ��@     T�@     B�@     J�@     �@     ��@     ��@     ��@     ��@     ��@     \�@     ݼ@     ��@     c�@     ӿ@     $�@     �@     c�@     l�@     ��@     ��@     \�@     {�@     ��@     3�@     ��@     ��@    �j�@    ���@     ٿ@     o�@     �@     �@     ��@     ��@     1�@     ٰ@     �@     ��@     ��@     ��@     ƥ@     �@     r�@     ȡ@     :�@     0�@     �@     ��@     ̙@     �@     |�@     ȕ@     ��@     �@     �@     �@      �@     p�@     4�@      �@     ��@     8�@     ��@     �@     ��@     ȇ@     ��@     ��@     ��@     ��@     ��@     ��@     P�@     �@     ��@     8�@     �~@     ��@     �@     �@     p}@     `|@     ��@     �y@      z@     �{@     �{@     Pz@     �w@     `x@     Px@     @x@     �s@     �s@     pu@      z@     �v@     0s@     r@     �r@     �q@     �s@     �p@     �p@     0p@     �o@     `m@     �l@     @n@     �l@     �m@     @k@     `i@     �k@     `h@     �f@     �g@     �e@      e@     �d@     �b@     �e@     �d@     �e@     �c@     @f@     @b@     �c@      `@      `@      a@     ``@     �\@     �a@     �`@     @a@     �a@      \@     �\@     @Y@      Y@     �[@      [@     �X@     �X@     @W@     �V@     �W@     �V@      Z@     �V@     �W@     @V@      [@     �R@     �Y@     �Z@     `p@     �Z@     �S@     �R@     �S@     �T@     �N@     �U@      R@     �Q@     �M@      O@     �P@      P@     �K@     �K@     �J@      P@      P@     �L@      G@     �G@     �G@     �N@     �K@      M@      I@     �E@      E@     �K@      L@     �F@      G@     �L@     �I@      H@      A@     �A@      E@     �H@     �E@     �C@      >@      B@     �E@      C@      B@     �C@      F@     �D@      <@      D@      C@      E@      6@      7@      B@      <@      5@      @@      =@      >@      :@      8@     �A@      ;@      @@      <@      ;@     �@@      :@      3@      8@      4@      6@      7@      <@      5@      4@      4@      >@      3@      1@      1@      7@      6@      3@      4@      .@      4@      7@      2@      1@      7@      0@      $@      @      *@      (@      &@      *@      (@      *@      (@      &@      *@       @      (@      (@      (@      ,@      *@      $@      @      &@      @      *@      $@     @s@      v@      @      @      @      @      @      �?      $@       @      &@      @      @      @       @      *@      &@      "@      "@      &@      ,@      $@      &@      0@       @      0@      @      "@       @      1@      &@      *@      $@      "@      @      .@      $@      ,@      2@      7@      6@      1@      $@      0@      0@      5@      ,@      6@      .@      5@      0@      .@      5@      >@      9@      ,@      6@      1@      3@      >@      5@      2@      ;@      @@      0@      8@     �@@      7@      :@      :@      9@      6@      ;@     �@@      @@      @@      <@      @@      A@     �A@      E@      B@     �C@     �E@      D@      8@      D@      J@     �F@     �A@      B@      L@      I@      E@      I@     �J@      H@      K@      H@     �E@      L@      M@      N@      D@      M@     �J@      O@     �P@     �R@      I@     �O@     �K@     �P@     @S@     @S@      S@     �S@     �S@     @Q@     @Q@      O@     �R@      R@     @Y@     �U@     @T@     @X@     @X@      W@     �Y@     �Y@     @Y@     �[@     �Y@     �Z@      Y@     �Y@     @^@     @[@     �Z@     �[@     @^@      [@      [@     �b@     �`@     �b@     �a@     �b@     �b@     �`@     �b@     @c@     �^@     `a@     `e@     �d@      b@      b@      u@      e@     @f@      f@     �s@     @j@     `h@     `i@     `j@      j@      k@     @j@     �k@     `l@     �m@      n@     �q@      u@     �p@     0q@     q@     @s@     Pr@     @r@      r@     �t@     ps@     �r@     �s@     �t@     �t@     �t@     �w@     `w@     �y@     �x@      z@     �w@     �z@     �|@     0}@     P~@     �@     P�@      �@     ��@     ؂@     ��@     ��@     ��@     ��@     �@     P�@     P�@     ��@     ��@     �@     ��@     �@     x�@     ��@     d�@     ��@     ؓ@     ��@     H�@     ܖ@     �@     d�@     �@     ��@     ��@     ��@     ޡ@      �@     >�@     &�@     N�@     b�@     �@     ֬@     ��@     ��@     Ų@     6�@     ��@     �@     һ@     R�@    �T�@     ��@     .�@    ���@    �.�@     g�@     ��@     ��@     j�@     ]�@     ��@     �@     ��@     ��@     �@     �@     ý@     ;�@     ��@     @�@     j�@     ^�@     =�@     ��@     �@     0�@      �@     ��@     ��@     ت@     �@     Z�@     j�@     �@     x�@     Ƭ@     ��@     �@     ��@     6�@     X�@     �@     گ@     �@     �@     p{@     @r@     �k@      _@     �R@      ;@      0@      @      @        
�
predictions*�	   ��Ͽ   �j@     ί@! �sN��P�)�D:�;@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��.����ڋ���(��澢f����pz�w�7�>I��P=�>f�ʜ�7
?>h�'�?�5�i}1?�T7��?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?uܬ�@8?��%>��:?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?�P�1���?3?��|�?w`<f@�6v��@�������:�               @      �?      �?      �?      @      @      &@      ,@      8@      7@     �A@      M@     �U@     �R@      \@     �^@     @^@      b@     `c@     �e@     �h@     �c@      d@     @c@      a@      c@     �`@     �W@      U@     @T@     �O@     @Q@      I@      H@      =@      @@      >@      >@      :@      9@      3@      *@      8@      ,@      *@      "@      (@      "@      @      @      @      @      @      @      @      @      �?      @      @       @      @      �?              �?       @      �?      @      �?      �?      �?      �?              @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @       @      @              �?      �?      �?      @      @      @       @      @      @      @      @      @      @      $@      @      @      @      "@       @      @      3@      ,@      3@      .@      ;@      5@      .@      9@      2@      8@      3@     �B@      6@     �B@     �B@      A@     �@@      C@     �B@     �D@      6@      <@      7@      ;@      1@      4@      ,@      6@       @      *@      0@      "@       @      @       @      @      @      "@      @      @       @      @      @      @       @       @      @      @      @      �?       @       @               @              �?       @      �?              �?        u yp3      ��<	�M|���A*�f

mean squared error�=

	r-squared0��>
�L
states*�L	   ����   `��@    �&A!D���@)�A��/��@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%               @      �?       @      @      @      4@      @@      L@     �W@     �d@     �|@     �@     x�@     n�@     +�@     �@     ̧@     ��@     $�@     R�@     ��@     �@     &�@     (�@     0�@     V�@     �@     �@     �@     T�@     ��@     ��@     5�@     ��@     �@     ǵ@     �@     �@     �@     �@     ��@     ��@     .�@     .�@     �@     �@     K�@     �@     >�@     �@     ��@     M�@    �S�@     A�@     �@     �@     u�@     ��@     ��@     ƶ@     g�@     5�@     ñ@     �@     T�@     �@     4�@     ��@     Ԥ@     ��@     6�@     l�@     ��@     \�@     0�@     ��@     �@     ؖ@     ��@     L�@     ؓ@     t�@     ��@     ��@     @�@     8�@     X�@     ؍@     X�@     ��@     8�@     ��@     ��@     H�@     ��@     `�@      �@     ȃ@     ؂@     (�@     h�@      �@     h�@     �}@     �@     �@     �~@      {@     ؁@     �@     �}@     �|@     y@     y@     �x@     @@     �w@     `w@      v@     �u@     �t@     ps@     pt@     �s@     �s@     ps@     �s@     @q@      o@     �p@      p@     q@     �n@     @o@      l@     �l@     �m@     @n@     �l@     @l@     �i@     �j@     �j@      f@     �g@     �i@     �f@     �e@     `c@     �d@     @f@     �d@     @e@     `e@     �b@     �c@      c@      a@      d@      a@      _@     @`@     @_@     ``@     �_@     @_@     @X@      \@      Y@     @W@      W@      V@      W@     �[@     �V@     �V@     @Y@     @U@      V@     �V@     @V@     @[@     �a@     @[@      R@     �S@      V@     �O@     �N@     �R@     �P@     @R@     @R@     �R@     @P@      M@      L@      L@     @P@     �F@      P@     �G@     �P@     �Q@     �g@     �P@      G@     @P@     �F@     �H@     �E@      L@      H@      F@     �G@     �C@      N@      J@     �E@     �E@      F@     �G@     �D@     �A@     �H@     �H@     �E@     �G@      C@      ?@      C@     �C@     �D@      F@      D@      D@      @@      :@      :@     �@@      =@      :@     �B@      B@      H@      :@      ;@      B@      A@      E@      >@      ;@      6@      <@      =@      ;@      >@      9@      8@      6@      5@      8@      9@      ?@      @@      5@      4@      9@      1@      6@      4@      ;@      7@      4@      5@      .@      ,@      6@      4@      1@      4@      6@      4@      ,@      ,@      2@      0@      3@      (@      (@      ,@      4@      ,@      *@      .@      ,@      *@      &@      (@       @      @     ��@     `w@       @      @      @       @      @      "@      @      @      @      "@      @      @      @      *@      @      @      @      &@      "@      0@      @      $@      "@      .@      &@      @      @      &@      0@      *@      .@      1@      *@      &@      ,@      0@      .@      (@      (@      "@      2@      &@      ,@      (@      *@      *@      >@      5@      5@      3@      3@      3@      ,@      3@      4@      5@      4@      6@      8@      5@      >@      ;@      7@      >@      5@      9@      ;@      :@      A@     �A@      8@      5@      F@      >@     �D@      <@      <@      @@      =@     �A@      D@     �B@     �B@     �I@     �C@     �B@      D@     �C@     �J@      I@      G@      C@      F@      G@     �G@      F@      C@      L@      L@      L@     �O@      K@     @Q@     �Q@      L@      J@     �O@     �Q@     @Q@     �R@     �N@     �N@     �P@     @T@      K@     �R@     �V@     �R@     @T@     �S@     �T@      V@     �Y@     �X@      R@      W@     �X@     �T@      R@     �Y@     �^@     �Z@      \@      \@      `@     �Z@     @Z@     �^@     @a@      `@      ]@     �`@      b@      ^@     ``@      b@     �`@     �`@     �b@     �`@      c@      c@     @f@      d@     �c@     @g@     @c@      c@     �e@     �i@     P{@     �g@      h@      g@     �i@     �k@      i@     @h@      n@     �m@     �n@     �m@     �n@     �j@     q@     �y@     �p@     �m@     Pp@     �n@     `o@     Pr@     �q@     �r@     �s@     �t@     @u@      u@      w@     �v@     �w@     �v@     �y@     �~@     �{@     �}@     �|@     �~@     @@     0�@      �@     ��@     ��@     �@      �@     ��@     0�@     ��@     ��@     �@     X�@     8�@      �@     ��@     8�@     X�@     D�@     �@     p�@     <�@     ��@     ��@     �@      �@     l�@     D�@     ��@     �@     ܡ@     N�@     F�@     f�@     :�@     �@     >�@     �@     ��@     �@     �@     ��@     �@     ��@     !�@     ��@     v�@     ��@    �^�@    ���@    �O�@     -�@     �@     h�@     ��@     ��@     �@     6�@     �@     ǽ@     ̽@     *�@     ��@     )�@     ��@     F�@     Q�@     ��@     H�@     ��@     S�@     b�@     ��@      �@     ��@     �@     ��@     ث@     ��@     �@     R�@     Z�@     ��@     ��@     �@     ��@     ��@     �@     L�@     ��@     x�@     ��@     D�@     (�@     �y@     @j@     @Q@      F@      ?@      0@      ,@      @      @      �?        
�
predictions*�	   ���ɿ    -�@     ί@!  ��B(@)4^G:�C@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.����ڋ������6�]���pz�w�7��})�l a���~���>�XQ��>�ߊ4F��>})�l a�>�FF�G ?��[�?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@�������:�              �?      @      @       @      @      *@      4@      4@      1@      @@     �G@     �J@      O@     �P@     @S@     �Q@      W@      a@      `@     �_@      d@     �_@      `@     �Z@     @]@      [@     �X@     @W@     @R@      O@     �Q@     @P@     �J@      B@     �B@     �A@     �B@      7@      5@      ?@     �@@      5@      2@      3@      0@      .@      @      &@      $@       @      @      @      "@      "@      "@       @      @       @      @              �?      �?      �?      @      @       @      �?      @      �?              �?       @      �?       @              �?      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?      @               @      �?              �?      �?      �?      �?      �?      @      �?      @      @       @      @      @      �?      @              @      @      @       @       @      "@      (@      @      &@      0@      $@      $@      0@      ,@      2@      5@      <@      .@      7@      6@      5@      ?@      >@      @@      B@     �@@      C@     �C@     �C@     �D@      ?@      D@     �D@      E@      D@      D@     �E@     �@@      >@     �@@      =@      ;@      <@      8@      ;@      7@      1@      7@      1@      *@      (@      3@      $@      "@      @      @      $@      @      @      @      @      @      @       @      @       @       @      �?      �?               @              �?              �?              �?        '�|�3      ���+	�]|���A*�e

mean squared error�D=

	r-squaredPU>
�L
states*�L	   ����   @��@    �&A!�;���@)g�:R��@2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              �?      �?              �?      @      3@     �A@      D@      O@      f@     `r@     ��@     r�@     �@     V�@     
�@     Щ@     ��@     (�@     �@     �@     إ@     *�@     ħ@     D�@     ة@     `�@      �@     �@     ��@     �@     �@      �@     ��@     '�@     N�@     ��@    �Q�@     ]�@     ��@    �o�@    ���@     C�@     ��@     ~�@     f�@     ��@     �@     C�@     ƹ@     D�@     s�@    ���@     #�@     ��@    ���@     a�@     k�@     >�@     �@     �@     )�@     r�@     h�@     v�@     ~�@     ҡ@     ��@     �@     ě@     0�@     ��@     ��@     ��@     t�@     �@     �@     p�@     Б@     Đ@     0�@     �@     Ќ@     �@     �@     ȍ@     ��@     h�@     H�@     �@     ��@      �@     ��@     ��@     �@     @�@     ��@     ��@     ��@     ؁@      �@     P~@     �~@     �|@     ~@     @�@     0�@     {@      }@     y@      y@     p{@     �z@      {@      x@      x@     �v@     0w@      ~@      z@     �z@     @w@     �v@      u@     �r@     �t@     �p@     @r@     `q@     �q@     �p@     0r@     �n@     Pq@     �o@     @n@      p@     �p@     �k@     �o@      g@      l@     �g@      k@     `j@      f@     @i@      f@     �f@     �d@     �h@     �c@     �e@     `d@     �b@      b@     `c@     @d@     �`@     �]@      `@     ``@     �^@     �^@      a@     �`@     @]@     �_@      \@     �Z@     �X@     �Z@     @\@     �[@     @Z@     �S@     �X@     �W@      V@      Y@     �S@     �V@     �^@     �T@     �V@     �[@     �\@      T@     �U@     @R@     �R@     �P@     �Q@     �O@     @P@     �T@      N@     �R@     �P@     @Q@     �M@     �N@     �K@      K@      M@      K@      O@      Q@      I@     �F@     �B@      H@     �F@      F@     �K@      N@      G@     �D@     �L@     �E@      F@     �B@     �H@     `b@     �F@     �L@      D@     �B@      H@      ?@      A@      E@     �@@     �D@     �C@     �B@     �E@      9@      =@      E@     �C@      6@     �B@      @@     �B@      ?@      ;@      <@      @@      =@     �@@      6@      :@     �@@      >@      7@      8@      :@      :@      A@      ?@      <@      6@      1@      7@      7@      7@      >@      4@      A@      @@      2@     �@@      9@      2@      4@      0@      7@      3@      .@      7@      3@      ;@      1@      0@      1@      6@      .@      3@      4@      2@      .@      (@      2@      ,@      ,@      ,@      $@      4@      0@      2@      ,@     ��@     p}@      @      &@      @      @      @      @      @      &@      (@      &@       @       @      $@      @      (@      "@      .@      ,@      3@      *@      $@      @      $@      (@      "@      $@      1@      3@      "@      .@      2@      (@      7@      4@      .@      1@      3@      2@      >@      5@      &@      (@      $@      9@      ,@      5@      :@      1@      8@      2@      @@      8@      4@      0@     �@@      5@      4@      7@      =@     �@@      ;@      >@      8@      ?@      >@      4@      B@      ?@      8@      @@      A@      @@      >@      8@      6@     �D@     �B@      F@     �H@      B@     �B@      C@     �B@      A@     �F@     �C@      K@      E@      I@     �P@     �H@     �F@     �P@     �N@      L@     �M@      J@      L@      M@     �K@      O@     �Q@     �N@     �P@     �P@     �V@      N@     �O@     @S@     �P@     @Q@     �P@      T@     �S@     �R@     �U@     �T@     @X@     �T@     �X@     @X@      Y@     @U@     �Y@     �U@     @X@      Y@     �X@     �\@     �[@     @X@     @Z@     @U@      [@     �]@     �Z@     �^@      a@     �a@     @`@     �c@      a@     �]@     @c@     �`@      b@      _@     �_@      d@     `a@      f@      b@      f@     @c@      f@     �e@     `e@     �j@     `j@      g@      j@     �k@      k@      n@     (�@      n@     @i@     �m@     @j@     �l@     @n@     �n@     @n@      p@     Pp@     q@      p@     r@     �s@     �x@     pr@     q@     �r@     `u@      v@     @v@     `v@      v@     �w@     �x@     �u@     @y@     �z@     �x@     �{@      @     P~@     �~@     (�@     p}@     Ѐ@     p�@     ��@     `�@     0�@     �@     ��@     �@     ��@     x�@     �@     ��@     ��@     H�@     ��@     x�@     ��@     ��@     ��@     ��@     <�@      �@     8�@     ��@     8�@     Ԕ@     |�@     \�@     ��@     ,�@      �@     ��@     ��@      @     ��@     N�@     ֧@     ��@     ��@     V�@     [�@     A�@     ��@     �@     k�@     ʽ@     ʿ@     U�@    ��@    ��@    �p�@     ��@     4�@     ߸@      �@     s�@     ��@     ��@     &�@     �@     Y�@     5�@    �#�@    �=�@    �C�@     d�@     +�@     n�@     �@     t�@     ��@     ��@     5�@     ��@     ��@     �@      �@     �@     t�@     �@     4�@     �@     ��@     �@      �@     p�@     L�@     �@     ı@     ��@     ��@     ��@     �@     0�@     P�@     �y@     @s@      `@      A@      7@      $@        
�
predictions*�	   ��.տ   @J~ @     ί@!  Э!�T�)&#<=@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x��>�?�s���O�ʗ����f����>��(���>a�Ϭ(�>8K�ߝ�>})�l a�>pz�w�7�>��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �?      �?              �?      �?              @      @      &@      2@      1@      ;@      @@      E@     �G@     �R@     @S@     @\@     �\@      d@     �f@     �d@     �k@     @k@     �f@      e@     @d@     `f@     �`@     �X@     �\@     �T@     �T@     @P@     @P@     �I@      F@      G@     �B@      >@      A@      2@      <@      7@      6@      8@      2@      *@      $@      4@      @      @      &@      @      @      @      @       @      @       @      @      @      @      @              @      @       @      �?      �?              @              �?              �?              �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?      @              �?              �?      �?       @      �?      @      �?      @       @      �?       @              @      @      @      @      @      @      @       @       @      "@      @      @       @      &@      "@       @      &@      "@      "@      $@      $@      *@      5@      .@      0@      1@      :@      5@      (@      5@      0@      .@      @      6@      6@      ,@      0@      ,@      &@      0@      ,@      ,@      *@      &@      $@      *@      &@      2@      ,@      2@      @      @      $@      @      @      @      @      @       @       @      �?      �?      �?       @      �?      �?      �?      �?              �?               @       @              �?              �?        F���2      ��ٞ	�%n|���A*�d

mean squared errorL=

	r-squared�[[>
�L
states*�L	   @���   ��a@    �&A!)7��@)�O�g�g�@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%               @      @      �?      &@      :@     �C@      K@      Z@      j@     �y@      �@     ��@     ��@     @�@     ��@     ��@     Χ@     �@     �@     ��@     �@     �@     �@     ��@      �@     *�@     R�@     Ϊ@     <�@     z�@     r�@     ~�@     -�@     �@     ˲@     ��@     |�@     M�@     %�@     �@     h�@    �/�@    ��@    ���@     �@     `�@     y�@     �@     -�@     ޵@     ˴@     ˴@     �@     ޵@     ��@     ��@     ��@     λ@     2�@     ��@     �@     ��@     X�@     ��@     ˱@     ��@     �@     �@     Ʀ@     l�@     b�@     ��@     ��@     �@     t�@     ę@     x�@     (�@      �@     h�@     �@     ܒ@     4�@     T�@     ��@     ��@     ��@     p�@     0�@     ��@     h�@     ��@     ��@     `�@      �@     ��@     (�@     ��@     ��@      �@     �@     ��@     H�@     ؀@     p@     �@     `�@     X�@     �}@     `}@     `}@     �}@     @}@     �x@     @z@     �z@      �@     ȁ@     p{@     �v@      w@     Pu@      v@     p}@     pt@     pr@     pt@     �s@     �q@     �p@     �p@     �p@     �q@     �p@      r@     `p@      o@     �l@     `n@     �k@     `l@     `l@     @n@      j@     �h@     �h@     �j@     �f@     �g@     �h@     �h@     `g@     �g@      c@     �g@     @c@      `@      d@     �c@      b@     �b@      e@     �^@     �a@     �c@     �^@     �_@     �]@     �\@     �b@     @Z@     @[@     �[@      _@     �\@     @[@     �[@     �V@     @W@     �W@     @Y@     �X@     �U@     �X@     �a@     @[@     �U@     �U@     �P@     �U@     �R@     �T@     �Q@     �T@     �P@      Q@     �Q@     �I@      S@      I@     @Q@      N@      M@      M@      E@     �K@      M@      I@      J@     �G@      H@     �P@      I@      B@      B@      G@      E@      G@      G@     �K@      D@     �F@     �B@     �H@     �e@      H@     �G@      J@     �B@     �G@     �C@      @@      D@      @@      D@      E@     �B@      C@     �E@     �B@     �B@     �D@      C@      F@     �A@      <@      D@      @@     �B@      =@      8@      :@      1@     �C@      ?@      B@     �@@     �@@      >@      8@      :@     �@@      ;@      2@      8@      9@      8@      9@      :@      4@      ?@      7@      <@      9@      2@      1@      9@      5@      7@      2@      ,@      (@      ,@      6@      6@      4@      4@      0@      3@      .@      2@      4@      1@      *@      0@      0@      ,@      4@      ,@      1@      *@      �@     �@      &@      "@      @      @      ,@      $@      @       @      @      *@      @      @      1@      "@      "@      1@      0@      "@       @      "@      *@      3@      &@      (@      @      (@      ,@      .@      2@      ,@      4@      2@      7@      0@      ,@      3@      0@      .@      3@      2@      0@      $@      &@      3@      4@      8@      5@      5@      7@      :@      (@      1@      5@      7@      :@      ;@      ;@      8@      9@      9@      9@      :@      F@      :@      E@      B@      :@     �B@      C@     �@@      A@     �A@     �C@      @@      @@     �A@      E@     �D@     �B@      A@     �E@      @@     �@@      J@     �D@     �B@     �F@     �E@      E@     �D@     �G@      G@     �J@     �I@      I@     �P@      I@     �G@      K@     �N@     �H@     �L@     �Q@     �Q@     �S@     �P@     @R@      T@     �O@     �R@     �O@     �R@     @T@      T@      X@     �T@     @Y@     �W@     @W@      X@     @V@     �T@     �W@     @[@     �Y@      ]@      \@      [@      ]@     �V@      `@      _@     @X@     �]@     �\@     �^@      `@     �^@      _@     �_@     `c@     �`@     �a@     @`@     �a@     �c@     @c@     `a@      c@      e@     �c@     @b@     �c@     �f@     �d@      d@     �g@     @h@     �h@     `i@     �i@     �h@     �g@     @k@     `l@      m@     �x@     �p@      n@      m@     �l@     �n@     @m@      q@     `q@     0q@     �p@     pt@      r@     �@      y@     �s@     �t@     0u@     Pu@     Pu@     �v@     �w@      x@     0w@     �x@     �z@     Py@     �|@      @     �}@     |@     ~@     �}@     ��@     ��@     X�@     @�@     x�@     ؃@     ��@     @�@     �@     ��@     h�@     �@     8�@     0�@     P�@     ��@     ��@     ��@     �@     4�@     X�@     ܒ@     $�@     0�@      �@     ��@     \�@     L�@     <�@     �@     $�@     B�@     ��@     f�@     ̨@     x�@     ��@     +�@     d�@     t�@     �@     t�@     ,�@     �@     Ի@     ��@     t�@     ��@     2�@     ��@     �@     ҵ@     �@     ��@     O�@     �@     =�@     ��@     ��@     D�@     ¿@    ��@     ^�@     ��@     T�@     ��@     $�@     ��@     ֲ@     �@     �@     $�@     ��@     n�@     s�@     B�@     ��@     ��@     8�@     ƭ@     #�@     F�@     ��@     r�@     X�@     d�@     M�@     ɼ@     J�@     ��@    ���@     @�@     x�@     ��@     ��@     ��@     ��@     �|@     @{@     �o@     �H@      ;@       @        
�
predictions*�	   @y�ο   �O}@     ί@!  ���\@)��[�`>@@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.��x?�x��>h�'��pz�w�7��})�l a�
�/eq
Ⱦ����žBvŐ�r>�H5�8�t>>h�'�?x?�x�?�[^:��"?U�4@@�$?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?w`<f@�6v��@�������:�              �?       @      �?       @       @      @      @      @       @      @       @      *@      *@      5@      9@      =@      7@      6@      5@      3@      4@      8@      :@      8@      3@      7@      9@      ;@      6@      3@      4@      0@      2@      8@      .@      3@      .@      "@      *@      $@      1@      @      *@      @       @      @      $@      @      @      @      @      @      @      @      @       @      @       @       @      @       @              �?               @      �?       @              �?      �?      �?      �?      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?              �?      �?       @              @      �?       @      @       @       @       @      @      @      @      @      @      @      @      @      "@      $@       @      ,@      2@      "@      1@      *@      .@      .@      :@      B@      <@      H@      C@      F@      I@      Q@     @R@     @T@     @_@     �W@     �a@     �d@     @c@     `e@     �i@      j@     �k@     �j@     `d@     �a@      [@     @T@     @R@     �E@      A@      A@      <@      @@      8@      3@      0@      *@      (@      *@      $@      .@      "@      @      @      @      @       @       @       @       @      @      @      �?      �?      �?      �?      �?               @               @              �?        y���3      �gw	��|���A*�g

mean squared error��=

	r-squaredL^>
�L
states*�L	   ��_�   @؟@    �&A!�����@)�
=qT��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @               @       @       @      4@     �C@     �S@     �c@     0p@     �}@     ��@     ڳ@     ��@     V�@     ,�@     (�@     ��@     Ҥ@     |�@     J�@     ʨ@     ��@     t�@     �@     `�@     �@     ��@     |�@     �@     ��@     ެ@     �@     ��@     �@     *�@     ��@     ��@     �@     ��@     �@     ʽ@     R�@     %�@    ��@     g�@     Ž@     ʼ@     ��@     ��@     M�@     ߶@     a�@     Y�@     ۷@     Ҹ@     t�@     º@     ��@     Y�@     .�@     >�@     �@     <�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     D�@     ؜@     ��@     H�@     t�@     ��@     ��@     �@     ��@     �@     �@     p�@     ��@     p�@      �@     Ќ@     x�@     ��@     `�@     H�@     ��@     ��@     �@     ��@     �@     ��@      �@     ��@     ��@     ��@     ��@     @�@     Ȁ@     P@     ��@     h�@      ~@     �@     ��@      �@     �y@     �y@     �{@      }@     �y@     �z@      w@      x@     �v@     0~@     pz@     �v@     �u@     �s@     �s@     s@     �s@     �z@     �r@     �q@     `r@     �q@     �n@     �m@     0p@     �p@     �o@     @m@     �p@     �w@     pp@     �k@     �m@      i@     �k@     @j@     �i@     `i@      d@     �g@     �f@     �e@      e@     �g@      e@     �c@      c@     �c@      d@     `c@     �b@      b@      e@     �b@     @b@     @^@      a@     @e@     �b@     @b@     �_@     �^@     �]@     �^@     @\@      ]@     �\@     �Z@      W@     �Z@      ]@     �^@      \@     �d@     @X@     �X@     �W@     �X@     �S@      V@     �S@     �Y@      R@     �V@     @V@     �S@      P@     @P@      O@     �S@     @P@     �S@     �P@     @Q@     �I@      L@     �D@     �N@     �G@     �P@      M@     �J@      I@      I@      K@      L@      F@     �N@      K@      F@      H@      M@     �J@     �K@      D@     �C@     �H@     �F@      f@      M@     �F@     �D@      ?@      H@     �B@      C@     �A@     �@@      >@      F@     �B@      D@     �B@      G@      =@      ;@      E@      @@      ?@      B@      C@      @@      =@      7@      A@      @@     �A@      :@     �A@      6@      C@     �A@      8@      =@      5@      >@      :@      2@      ?@      1@      6@      :@      3@      3@      :@      ;@      2@      4@      ;@      2@      ?@     �B@      7@      3@      *@      ?@      4@      3@      3@      "@      6@      0@      *@      @      ,@      *@      3@      ,@       @     �@     ��@      (@       @       @      (@      $@      @      "@      "@      *@      (@      ,@      ,@      *@      (@      0@      @      "@      .@      *@      .@      1@      .@      *@      ,@      *@      $@      *@      @      0@      *@      1@      .@      2@      (@      (@      "@      4@      1@      :@      3@      *@      3@      :@      3@      0@      5@      6@      0@      ;@      :@      :@      9@      =@      8@      4@      7@      7@      <@      6@      6@      5@      ?@      =@      @@     �G@      ?@     �B@      ?@     �C@      8@      >@      =@      ;@      A@      >@     �C@     �J@      >@      ?@     �C@     �I@      C@      B@     �@@     �G@     �D@     �D@      K@     �C@     �G@      P@     �C@      K@      D@      Q@     �O@      J@      M@      O@     �M@     �P@     �K@     �Q@     �P@      I@     @T@      P@     �P@     @S@     �T@     �R@     �U@     @S@     �Q@      S@      T@     �S@     @V@     �W@     @V@      Y@     �W@     @Y@     �U@     �Y@      \@     �]@     �X@     @\@      _@     �X@     @_@     @[@     �[@      \@      Y@      b@     @^@     @\@      d@     �_@      _@     `c@     �c@     �b@     �b@      c@     �b@     �b@     `b@     �b@     @d@     �c@     `b@      g@     �l@      k@      h@      k@     @g@     `j@     `l@     �i@     �l@      i@     �n@      m@     �j@     �w@     �q@     �m@     `q@     p@      s@     0r@     �o@     �r@     s@     @q@     �r@     �t@     t@     �u@      v@      v@     pu@     (�@     0z@     v@     Py@     `z@      z@     `~@     �}@     �z@     @}@     �}@     �~@     H�@     ��@     ��@     P�@     x�@     ��@     ��@     ��@     x�@     ��@     p�@     ��@     ��@     ؆@     ؉@     �@     �@     ��@     X�@     `�@     ��@     Ē@     `�@     p�@     ��@     ,�@     \�@     ��@     �@     @�@     ��@     d�@     ��@     ~�@     ��@     ҥ@     ��@     ҫ@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ټ@     p�@     �@     �@     Z�@      �@     ��@     �@     ߷@     ո@     �@     N�@     ��@     Ҿ@     �@     �@     ��@     ؿ@     u�@     ��@     ��@     ��@     |�@     n�@     �@     ��@     �@     2�@     ��@     I�@     |�@     '�@     t�@     ��@     ��@     ҭ@     ��@     ��@     `�@     Z�@     �@     ��@     �@      �@     �@     ��@    ��@     d�@     �@     ؍@     @�@     ��@     P�@     �~@     �z@     �y@      N@     �F@      @        
�
predictions*�	   @pѿ   ��@     ί@!  ��R4@)}���;@2�_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1�f�ʜ�7
������6�]���O�ʗ�����Zr[v���h���`�8K�ߝ뾢f�����uE������[�?1��a˲?6�]��?����?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?��tM@w`<f@�������:�              �?       @      @      @      ,@      (@      .@      :@      <@      :@     �B@     �@@     �@@      @@      B@      D@     �D@     �A@      <@      M@     �C@     �@@      F@      B@     �F@      F@      E@     �C@     �F@     �H@     �B@      F@      B@     �A@      :@      @@      B@      <@      =@      :@      1@      5@      2@      0@      :@      5@      &@      0@      *@      @      @      (@       @      &@      @      @      @      @      @      @      @       @      @      @      @       @      @       @      @      @      �?      @              @      �?       @              �?       @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @       @               @       @      �?               @      �?       @      @      �?      �?              �?      @      �?      �?      @      @      @      @      @      @       @      $@      $@      &@      $@      &@      3@      4@       @      3@      ;@      8@      A@      A@      D@      B@     �H@      M@     �E@      K@     �Q@      P@     �R@     @X@     @V@     @^@      [@     �]@     @\@     @b@      a@     @_@     �W@     �U@     @U@      M@     �D@     �C@      @@      6@      9@      6@      1@      "@      1@      $@      (@      5@      0@      @       @      @      @      @      @      @      �?      @      @      @      @       @               @       @       @       @      �?      �?      �?      �?              �?      �?              �?        I��3      Y2Y	'�|���A*�g

mean squared error>=

	r-squared�@�>
�L
states*�L	    ���    �{@    �&A!��	HX�@)���Dع�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �?       @              @      @      2@      1@      >@     �R@     @c@     Pw@     �@     -�@     `�@     $�@     ��@     �@     V�@     �@     �@     j�@     �@     
�@     T�@     ԥ@     ��@     ��@     ��@     ֥@     ��@     ��@     �@     �@     u�@     ;�@     g�@     `�@     ��@     ׸@     ��@     ��@     ��@    ��@     f�@    �"�@     M�@     ��@     D�@     ��@     ��@     8�@     �@     �@     �@     Ļ@     �@     ��@     տ@     �@     <�@     ��@     Ʒ@     "�@     `�@     c�@     �@     P�@     ܨ@     
�@     b�@     ��@     .�@     d�@     (�@     l�@     p�@     ��@     (�@     ��@     ��@     ��@     ��@     ��@     Đ@     �@     ؎@     ��@     ��@     ��@     x�@     p�@     x�@     ؉@     �@     0�@     ��@      �@      �@     �@     0�@     ��@     Ȃ@     ��@     �@     @�@     ��@     ��@     ��@     �|@     P~@     �@     0@     �y@     `z@     0�@      {@     �z@     0{@      z@     �x@     �y@     �w@     �w@     �u@     `u@     �v@     `y@     �y@     �t@     �u@     Pp@      q@     �r@      w@     �q@     s@     �p@     �o@     �o@      o@      s@     �o@      o@     �l@      k@      k@     �i@     �k@     �j@     �k@      g@     �g@     �h@     �e@     �g@      f@      d@     �e@      e@     �d@     �m@     @p@      c@     �d@      c@     @b@      c@      a@     @]@     `c@     @^@     @`@     @^@     �a@     @^@     �`@     @]@     `a@      `@     �a@     �e@      b@      `@      X@      W@     �U@     �X@     �U@     �W@     @Y@     @W@     �W@     @V@     �S@     �U@      Y@     �S@     �T@     �R@      U@     �S@     �S@     �Q@     �S@     �Y@     �P@     �Q@      T@     �P@     �M@      S@      Q@      M@      M@     �R@     �M@      L@     �Q@     @P@      E@     �M@      H@     �I@      L@      M@     �N@      C@     �B@      K@     �L@      F@     @S@      c@     �D@     �H@      J@     �G@     �C@      H@     �G@      A@      H@      D@      ;@     �B@      G@     �I@     �D@      =@     �B@     �B@     �C@     �B@     �E@     �I@     �C@      5@      :@      8@      A@      8@      =@     �C@      ?@      <@      =@      =@     �A@      =@      9@      >@      8@      6@      ?@      :@      8@     �C@      4@      5@      9@      0@      6@      6@      6@      4@      .@      :@      ,@      9@      8@      3@      4@      >@      (@      *@      4@      6@      &@      &@      (@      (@      6@      �@      �@      (@      "@      @      "@      ,@      $@      ,@      ,@      1@      "@      @      ,@      "@      ,@      &@      @      &@      (@      *@      1@      &@      &@      .@      0@      0@      (@      0@      3@      (@      ,@      3@      2@      ,@      (@      8@      4@      3@      3@      :@      7@      "@      6@      6@      5@      <@      5@      6@      7@      8@      9@      :@     �@@      D@      :@     �A@      ?@     �C@      =@     �A@     �B@     �A@      >@      B@      ?@      ?@      ?@      =@      ?@      =@      @@     �D@      C@      D@     �B@     �E@     �E@      G@     �A@     �E@      A@      H@      H@      H@      @@      J@      K@     �G@      G@      F@      L@      O@      O@     �G@      N@     �O@     �L@     �P@      Q@      M@     @P@     @Q@      R@     @S@     @S@     �O@      Q@      Q@     @T@     �T@     @U@     @V@     �S@      S@      T@      X@      S@      U@      \@     �U@     @X@      X@     �Z@     @_@     �Y@     @^@     �\@     �\@     �_@     @]@      _@     @[@      _@     �`@     @\@     �_@      b@     �a@     �`@     �^@     �_@     �a@     @e@     �b@     �c@     `d@     �b@     @a@      d@      e@     `c@      d@     `e@     �g@     `e@      j@      l@     �h@     �h@     �h@     `h@     �i@      k@     �j@      l@     �l@     �l@     �m@     �n@     0p@     �l@     `o@     �s@     �q@     {@     �q@      r@     �s@     Ps@     �r@     ps@     s@     Pt@     Pv@     �t@     x@     �~@     `x@     0y@     �w@     �x@     `z@     z@     `|@     ��@      }@     P}@     ��@     8�@     ؁@     ��@     X�@     �@     ��@     ��@     �@     ��@     Ѓ@     h�@     ��@     ��@     ��@     x�@     8�@     ��@     �@     p�@     `�@     ��@     �@     ��@     ��@     ,�@     ̔@     �@      �@     �@     L�@     d�@     ��@     h�@     �@     �@     <�@     x�@     ާ@     ,�@     ̫@     ��@     �@     [�@     �@     �@     r�@     t�@     �@     ��@     u�@     r�@     c�@     ĺ@     �@     =�@     ��@     �@     C�@     U�@     ��@     �@     ž@     �@     �@     ��@     ��@     Q�@     �@     ҵ@     �@     F�@     T�@     ְ@     /�@     F�@     ��@     �@     ȫ@     Z�@     �@     b�@     ��@     ��@     ��@     d�@     R�@     >�@     -�@     ��@     F�@     ?�@     .�@    �d�@     d�@     d�@     x�@     h�@     Ђ@     �@     �}@     �y@     �d@     �E@      A@      �?        
�
predictions*�	   �ۻп   �B|@     ί@!  ��:�)�$���EE@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r������6�]�����Zr[v��I��P=��pz�w�7��})�l a���(��澢f���侄iD*L��>E��a�W�>����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@h�5�@�Š)U	@�������:�               @      @      @       @      @      3@      1@      9@      <@      D@     �G@      C@     �O@     @P@      K@      O@     @P@      R@     �U@      X@     @U@     @Y@     �[@     �]@     �Y@      Z@     @Z@     �Z@     �X@     �U@      W@     @V@     �W@     �P@     @P@      L@     �P@      N@      B@      H@     �I@      D@      8@      4@      <@      5@      0@      7@      4@      9@      &@      ,@      "@      &@      "@      @      $@      @       @      @      @      @       @      @      @      @       @       @       @      �?               @      �?              @       @      �?               @       @      �?      �?               @      �?              �?               @              �?              �?              �?              �?               @              �?              �?       @              �?      �?      @               @      �?      @      @      @      �?       @      �?      @      @      @      @      @      @      @      "@       @       @       @      @      $@      @      $@      &@      "@      2@      9@      ,@      5@      .@      <@      7@      =@      @@      8@      4@      .@      9@      @@      8@      3@      >@      ?@      :@      ?@      ;@      8@      =@     �C@      1@      =@      7@      6@      4@      :@      :@      3@      <@      .@      5@      1@      (@      5@       @      $@      *@      @      $@      (@      @      @      @       @      $@       @      @       @      �?      @       @      @              @       @      �?      �?              �?              �?               @              �?        )�;�r3      �i	D��|���A*�f

mean squared error��=

	r-squared���>
�L
states*�L	    ���   @l�@    �&A!����q�@)I�!���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      �?       @       @      $@      >@      8@      C@      O@     @X@     �q@     (�@     ��@     "�@     @�@     !�@     ��@     �@     �@     0�@     �@     �@     F�@     ֢@     ��@     ܣ@     F�@     h�@     ��@     F�@     �@     ��@     |�@     �@     -�@     h�@     ų@     ��@     ͷ@     ܺ@     4�@     �@    �
�@    �f�@     �@     ��@     �@     �@     y�@     ^�@     p�@     ��@     �@     �@     �@     ��@     ��@     R�@     ��@     �@     ��@     G�@     ��@     ��@     ��@     9�@     ��@     z�@     ��@     V�@     ��@     ¢@     ��@     v�@     ԝ@     ��@     <�@     4�@     ��@     $�@     ��@     Г@     Ԓ@     �@     ��@     ��@     ��@     `�@     ��@     8�@      �@      �@     8�@     H�@     x�@     ��@      �@     ��@      �@     ��@     @�@     ȃ@     Ȃ@     ��@     P�@     ��@     `�@     ؄@     P�@     �|@     �@     P|@     �|@     {@     �{@     �z@     `@     @}@     �x@     �y@     �u@     @u@     �x@     �v@     `v@     �u@     �v@     �t@     @v@     �|@     �x@     �q@     �p@     0r@      q@     �p@      q@     �r@      r@     �n@     @l@     �o@      m@     �j@     �i@     `l@     �l@     @l@      l@     �k@     @i@     �i@     �e@      h@     �h@     `g@     �f@     @e@      e@      g@     �c@     �d@      e@      f@     @b@     `d@     �`@     �d@      c@      d@     �l@     �d@     �`@     �`@      a@     �^@     �_@      \@      \@     @a@     @a@     �e@     @[@      \@     �[@     @Z@     @Z@     �Z@     @\@     @U@     �X@     �V@     �Z@     �Y@     @[@     �R@     �T@     �S@      N@      T@     �R@     �S@     �Q@     @R@     �R@     @Q@     @S@     �Q@      O@     �T@     �R@     �P@     �T@     @P@     @R@     �P@     @P@      =@      P@      M@      M@     �L@     �M@      K@     �M@      P@     @f@      N@      C@      G@      H@     �K@     �N@      I@     �E@      A@     �F@     �I@     �E@     @P@     �H@      D@     �B@      C@     �D@      ?@     �B@      B@      I@      B@      C@      F@     �B@      5@     �B@      C@      A@      ?@      ;@      6@      >@      >@      A@      <@      <@      :@      7@      @@      <@      :@     �@@      6@      A@      A@      7@      6@      8@      4@      @@      2@      6@      <@      5@      <@      ;@      <@      7@      =@      *@      5@      6@      4@      ;@      1@      =@      .@      1@      3@      1@      ,@      5@      ,@     �@     X�@      (@      "@       @      .@      .@      @      0@      $@      "@      $@      *@      (@      5@      0@      .@      (@      &@      2@      (@      1@      (@      "@      0@      2@      .@      4@      &@      3@      2@      1@      3@      5@      1@      9@      3@      1@      <@      5@      .@      5@      2@      5@      3@      3@      >@      9@      7@      :@      @@      >@      2@      <@      ?@      8@      ?@      5@     �E@      <@      ?@      =@      ?@      B@      ;@      >@      >@      ?@      E@     �C@      A@     �C@     �A@     �F@      D@      N@      D@      @@     �E@      G@     �F@      E@     �I@      L@     �I@     @P@     �P@      I@      K@      R@     �C@     �N@     �P@     �L@      O@     �J@     �I@     @P@     �I@     @S@      P@     �O@     @Q@     �Q@     �S@     @R@      T@     @T@     �R@     �R@     �V@      T@     �S@     �W@     �Y@     �W@      Y@     @Z@      \@     �Y@     @Z@      V@     @Y@     �Z@     @^@     �[@      [@      \@      \@     �`@      a@     @]@      \@      [@     �`@     `d@     @`@     ``@     �\@      a@      a@     `b@      b@     @c@      b@      c@     �b@      e@     �e@     �c@     @c@      f@     �f@     @j@     @h@     �d@     �g@     `i@     `h@      j@     �l@     @k@     �l@      j@      m@     @k@      m@     `k@     `k@     pp@      n@     @o@     pq@     0z@     �u@     �s@     �p@     �p@     Pr@     �q@     �s@      t@     `u@     w@     �~@     t@     �t@     @v@      v@     �v@     �x@     y@     `z@     �|@     P{@      }@     �}@     h�@     `�@     H�@     P�@     ��@     �@     ��@     @�@     ��@     Ђ@     ؃@     ��@     ��@     Ѕ@     �@     ��@      �@      �@     ��@     ؐ@     8�@     (�@     ܐ@     h�@     ��@     �@     P�@     h�@     ��@     ��@     X�@     Ĝ@     ��@     X�@     0�@     B�@     l�@     ��@     ��@     ©@     ��@     �@     ��@     9�@     ��@     ̶@     ø@     �@     ]�@     ��@     _�@     >�@     ��@     e�@     �@     D�@     �@     ��@     0�@     �@     Q�@     ��@     ��@     k�@     ֿ@     ٿ@     ��@     Ͻ@     ֺ@     ��@     g�@     մ@     �@     ��@     ~�@     J�@     ��@     ��@     ҫ@     �@     ��@     ا@     ��@     (�@     ��@     Ĩ@     ��@     �@     ܬ@     �@     V�@     ؼ@     b�@     d�@     g�@     �@     @�@     H�@     ��@     �@     @�@     �~@     0r@     �E@      B@      6@      @        
�
predictions*�	    {ֿ    ��@     ί@!  ����1�)&��j�B@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ���d�r�x?�x��>h�'��f�ʜ�7
���[���FF�G �O�ʗ�����Zr[v��pz�w�7�>I��P=�>f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?yL�����?S�Fi��?ܔ�.�u�?�6v��@h�5�@�������:�              �?              �?               @      @      �?      @      @      "@      &@      (@      .@     �@@      @@      D@     �N@     �N@     �I@     @T@      W@     @V@     �[@      W@     �[@     �X@     �]@      \@     �\@     �X@     �]@     @X@     �V@     @S@     @X@     �S@     �R@      M@     �O@      F@      D@      I@     �C@      ?@     �@@     �@@      :@     �@@      4@      8@      <@      ,@       @      (@      "@      @      @      $@      (@       @      @      @       @      @      @      �?      @      @      @      �?      @      @      �?       @              �?       @              �?      �?       @               @      �?      �?              �?      �?      �?              �?              �?              �?              �?      �?              �?               @              �?      @      �?      @      �?      �?       @      @              �?       @      @      �?      @      @      @      @      "@       @      @      @      &@      @      "@      "@      (@      $@      *@      ,@      &@      4@      ,@      3@      >@      >@      6@      8@      9@      8@      5@      @@      E@     �E@     �E@      F@     �H@      =@      B@      ?@      6@     �D@      9@      =@      >@      8@      8@      6@      ;@      8@      *@      ;@      8@      8@      4@      *@      .@      "@      ,@      "@      2@      "@      @      @       @      *@      "@      @      @      @       @       @      @      @      @      �?      @               @      �?              �?       @      �?              �?       @              �?        Y�p��3      �=%�	�߳|���A*�g

mean squared errorl/=

	r-squaredx3�>
�L
states*�L	   `���    ��@    �&A!��@�ܫ�@)���J��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �?       @       @      @      2@     �C@      H@     @Q@     �_@     �c@     u@     ؄@     ��@     (�@     x�@     �@     �@     <�@     V�@     �@     ��@     ��@     �@     @�@     
�@     ԣ@     �@     ҥ@     ڦ@     ȥ@     ��@     ��@     ��@     Ʈ@     �@     �@     f�@     F�@     ��@     �@     ��@     ؽ@     ��@     ��@     ؼ@     ��@     z�@     '�@     ��@     �@     �@     ��@     �@     v�@     ]�@     �@     ��@     !�@     
�@     '�@     1�@     Ӹ@     �@     \�@     �@     	�@     4�@     ¬@      �@     V�@     ��@     L�@     |�@     �@     �@     <�@     0�@     ��@     ��@     ��@     ,�@     ��@     l�@     (�@     ��@     4�@     P�@     x�@     X�@     �@     ��@     �@     ��@     ؈@     ��@     0�@     ؆@     X�@     ��@     ȉ@     @�@     h�@     Ȃ@     ��@     X�@     ��@     0�@     p�@     ȁ@     �@     0�@     @�@     �@     P@      |@     �z@     �|@     p{@     �z@      z@     @}@      z@     `w@     �@     �z@     �u@     �w@     �t@     �x@     |@     Ps@     @t@     �t@     s@     �s@     �t@     r@     �q@     r@     �n@     q@     q@     �q@     �x@     �k@     `p@     �q@     �n@     �k@     �l@     �k@     `k@     �i@     �i@     `h@     `h@     @g@      i@     �g@     �e@     �f@      d@     `g@     @d@     �e@     �d@     @e@     �e@     �c@      b@      c@     �d@      c@     0q@     �a@     �c@      _@     `a@     �`@     �\@      a@     �Z@     �`@     @[@     �\@     �a@     �d@     �]@      Y@     �_@     @Z@      Y@     �X@      X@     @Z@      U@     �W@     �V@     @S@      R@     �U@     �R@     �P@     @T@      R@      O@      S@     �U@      W@      P@     �S@      L@      M@      h@     �R@     @Q@      J@     @R@     �Q@      O@      N@     �L@      Q@     �L@     �N@      C@      F@      Q@     �N@     �O@      H@      K@      J@     �L@     �J@      H@     �I@      E@     �I@      G@     �K@      E@      B@     �F@     �K@      @@     �D@     �B@     �E@     �C@     �F@      J@      9@     �A@     �F@      B@     �B@     �B@      F@      ?@      <@      A@     �A@      B@      B@     �@@      9@      9@      B@      :@      >@     �B@      9@     �A@     �A@      B@      9@      >@      3@      ?@      6@      =@      >@      3@     �@@      7@      .@      @@      4@      2@      0@      2@      6@      8@      0@      5@      7@      6@      3@      3@      5@      $@     H�@     H�@      1@      "@      $@      "@      .@       @      .@      0@      @      .@      &@      $@      0@      ,@      0@      5@      *@      2@      1@      4@      *@      ,@      ,@      .@      1@      .@      9@      .@      6@      9@      :@      2@      ;@      0@      :@      3@      9@      1@      0@      9@      8@      .@      =@      @@      9@      ;@      6@     �B@      >@      B@     �B@      ;@      9@      =@     �B@      <@      @@      9@     �C@     �B@      @@     �@@      >@     �D@      H@      A@     �B@      G@     �G@     �A@      A@     �E@     �D@     �H@     �K@     �K@     �G@      G@      L@     �H@      N@     �O@      J@      K@     �R@      O@     �N@     �P@     �L@     @P@      K@      R@     �O@     �T@      T@     �R@      M@     �Q@     @X@     �Q@     �S@     @S@     �Q@      U@     @W@     �U@     �Q@     �V@     @T@      Y@      Z@     �V@     �U@     �X@     @\@     �W@     @Y@     @\@     �W@      ]@     @\@     �[@     �\@     �`@     �a@      \@     �\@     �\@      _@      _@     �a@     �`@      `@     �b@      d@      ]@     �`@      b@      a@     �b@      f@     `c@     �c@      c@     `c@     @f@     �f@      f@     �h@     �i@      k@     �i@     �f@      g@     `i@      k@      i@     �g@      m@     �j@     �m@      p@     Pv@     �p@     @l@     `o@      m@     @p@     �o@     Pp@     �p@     �q@     �p@     �q@     �t@     �q@      s@     �r@     �x@     �w@     Pu@     �v@     �t@     pv@     �v@     Px@     Pw@     0v@     �x@     (�@     �|@      y@     �{@     }@     @�@     ȁ@     ��@     ؀@     ��@     ��@     H�@     ��@     ��@     ��@     �@     Ѓ@     ��@      �@     ��@     X�@     ��@     �@     Љ@     �@     ��@     Ȏ@     (�@     Б@     ��@     8�@     8�@     ��@      �@     �@     P�@     \�@     �@     ��@     Ҡ@     J�@     ��@     ��@     �@     ب@     (�@     8�@     ��@     v�@     ��@     ��@     �@     
�@     e�@     >�@     
�@     ߻@     ��@     X�@     {�@     x�@     ��@     ��@     ��@     ��@     o�@     ��@     ��@     t�@     ��@     k�@     i�@     �@     ��@     `�@     ^�@     R�@     1�@     ��@     ��@     ��@     Į@     ��@     ��@     ��@     ��@     �@     ��@     6�@     2�@     V�@     4�@     R�@     ��@     �@     ��@     ��@     �@     f�@     x�@     y�@      �@     ��@     ��@     p@      |@     `~@     �@      k@      F@      ?@      6@       @        
�
predictions*�	    ŝԿ   @�@     ί@!  ZiafC@)����D@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���>h�'��f�ʜ�7
������6�]�����Zr[v��I��P=��1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�6v��@h�5�@�������:�              �?              �?      �?       @      �?      @       @      @      "@      0@      2@      3@      ?@      >@      A@      B@      J@      L@      P@      P@     �Q@     �W@     �R@     �U@      V@     �R@      T@      Q@      Q@     @Q@      N@     @Q@     �Q@      O@      H@      E@      K@     �K@     �D@     �B@     �F@      ;@      7@      7@      ;@      ;@      3@      3@      (@      $@      $@      *@      @      &@      "@      @      &@      @      @      @       @       @      @      @      @      @      �?      �?      @      @      �?              �?      @      �?              �?      �?              �?              �?      @       @              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?       @       @      @      @               @      �?      �?      �?      @      @      @      �?      @      @      �?      @      @       @       @      @      @      "@      &@      "@      @      @      (@      @      $@      *@      0@      2@      6@      3@      9@      @@      ;@      7@      4@      E@      >@      ?@     �E@      E@     �K@      N@     �M@     �M@      P@      N@     �R@     �R@     �P@     �Q@      O@     �I@      K@     �H@     �D@      H@      D@      C@      >@     �B@      :@      ;@      =@      6@      0@      0@      2@      .@      4@      @      (@       @      @      ,@      "@       @      @      @      @              @       @       @               @       @       @              �?      �?              �?              �?        �Gxb3      ��v	%��|���A*�f

mean squared error �=

	r-squared�K�>
�L
states*�L	    '��   ��@    �&A!��n��@)��1&�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �?      @      @      ,@     �C@     �R@     �U@      Y@     �g@      m@      z@     8�@     ׺@     �@     (�@     ��@     ��@     Ȩ@     ��@     �@     ��@     ��@     z�@     ޢ@      �@     J�@     ��@     ,�@     ��@     �@     ��@     �@     ��@     5�@     ��@     Ӳ@     �@     |�@     �@     C�@     ɻ@     8�@     ��@     W�@     �@     ��@     K�@     f�@     ��@     ��@     ��@     l�@     ��@     ��@     �@     >�@     ��@     +�@     [�@     �@     ��@     ø@     ��@     ��@     4�@     ۰@     ��@     N�@     ,�@     �@     ��@     ̢@     N�@     `�@     ��@     T�@     ��@     З@     ��@     L�@     �@     X�@     H�@     ��@     �@     t�@     @�@     ��@     P�@     ��@      �@     H�@     ��@     ��@     X�@     ��@      �@     ؃@     ��@     ��@      �@     h�@     �@     0�@     ��@     (�@     �@     ��@     H�@      �@     ؀@     p@      ~@     �~@     @~@     `�@     �z@     p{@     �|@     0y@      {@     X�@     0x@     �x@     0{@     �w@     �v@     @x@     w@      v@      u@      u@     @t@      u@     �t@     �s@     �t@     �s@     0w@     0u@     Pp@      n@     `o@     �o@     �q@      o@     �p@      l@     �p@     �k@      o@     �i@     0p@     �q@     �m@     �h@      i@     @j@     @i@     @h@     �i@     `h@      h@     @f@     @i@     �h@     �p@     �i@      b@     �e@     �c@     �c@     �a@      b@      a@     @a@     �a@     @b@     ``@     �`@     �^@      `@      `@     �`@     �]@     ``@     @`@     @c@     �\@     �Y@     �U@      \@     �W@     �Y@     �W@     @Z@      U@     �W@     �V@     �U@     �S@      S@     @V@     @T@     �T@     �N@     �R@     �P@     �V@      T@      e@      S@      S@     �S@     �R@     �R@     @Q@     �Q@     �Q@     �M@     �Q@      R@     @Q@     �Q@     �C@      M@     @R@      K@     �F@      L@     �M@     �B@     �F@     �K@     �L@      H@     �J@     �M@      E@      D@      K@      C@      G@     �E@     �G@      F@     �E@     �F@     �A@      G@      F@      E@     �A@      >@     �C@     �D@      A@     �@@     �B@      >@      @@     �A@     �H@      C@      D@      A@     �G@      F@      ?@      8@      4@      C@      @@      >@     �@@      9@      <@      ?@      9@      8@      B@      7@      8@      8@      3@      0@      5@      :@      3@      <@      7@      6@      &@      7@      3@      4@      8@      6@      8@       @      5@      4@     (�@     @�@      *@      $@      @      6@      (@      .@       @      &@      $@      2@      2@      1@      .@      *@      ,@      ,@      2@      7@      4@      5@      4@      :@      8@      *@      5@      9@      8@      8@      0@      0@      (@      4@      =@      7@      9@      6@      8@      <@      9@     �@@      7@      D@      :@      A@      ;@      5@      =@      E@      3@     �D@     �A@      <@      A@      C@     �A@      :@     �B@      ?@     �B@      A@      E@     �A@     �G@     �C@      M@      L@     �B@      D@     �J@     �J@      H@     �J@      D@     �L@      M@     �M@     �J@      L@     �L@      M@     �J@     �H@     �N@     @P@     �Q@      S@     �P@     �Q@     �Q@     �O@     @Q@     �S@     �P@     @S@     �U@     �S@      Q@      T@     �W@     �S@     @S@     @W@     @S@      Y@     @S@     �V@     @Y@      V@     �W@     �Y@      [@     �Y@     �Y@     �Z@     @\@     �V@     �Z@     �^@     @]@      \@     �^@     @_@     ``@     @]@      ^@     @^@     �_@     @`@     �_@     �`@      c@     �a@     �b@     �c@     �_@     �b@      b@     �d@      f@     �f@     �c@     �c@     `e@     @d@     �f@     �c@     @g@     �f@     �g@     �e@     @g@     �e@     �i@      j@     �n@     t@     �p@     �l@     �j@     �l@     �i@     �n@     �n@      m@     �k@     �p@     0p@     �m@     �p@     `p@     �q@     0p@     �p@     �q@     @r@     �r@     �s@     �}@     �u@     `t@     `t@     `s@     �u@     `u@     0w@     �v@     0v@     `y@     0z@     Py@     P@     �@     �}@     P{@     ~@     ��@     �~@     ��@     (�@     �@     h�@     ��@     �@     ��@     x�@     ��@     Є@     ��@     ��@     h�@     ȇ@     ��@     ȉ@     h�@      �@      �@     �@     x�@     4�@     ��@     ��@     Ĕ@     $�@     �@     ��@     ,�@     ��@     l�@     (�@     N�@     \�@     ��@     ��@     ̧@     d�@     p�@     ��@     ?�@     i�@     �@     1�@     ��@     ��@     ��@     ��@     ��@     ��@     i�@     #�@     �@     s�@     �@     ;�@     b�@     �@     ú@     ڻ@     ��@     ��@     ?�@     ��@     q�@     �@     ��@     �@     3�@     u�@     =�@     ��@     #�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     �@     f�@     ��@     �@     �@     Ы@     ��@      �@     0�@     z�@     ��@     ĳ@    ��@     ,�@     H�@     ��@     �@     �{@     �|@     �{@     �g@      F@      1@      2@      @        
�
predictions*�	   ���Կ    3�@     ί@!  ��NB$�)�8K~�:L@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s����h���`�8K�ߝ�1��a˲?6�]��?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?u�rʭ�@�DK��@�������:�              �?      �?              @      @      $@      &@      *@      3@      6@      :@     �B@      F@     �J@     �J@      L@      K@     @R@      R@     �R@      U@      T@     �R@      W@     @U@      W@     �V@     �U@     @W@     �R@      R@     �T@      Q@      J@      M@     �K@     �C@     �H@     �H@      A@      ?@     �F@      >@      <@     �A@      @@      B@      3@      4@      .@      (@      1@      ,@      @      .@      $@      @      @      @      @      @      @      @      @      @      @       @      @      @      @      �?       @       @       @               @      �?      �?              �?      �?              �?              �?              �?      �?              �?              �?               @              �?      �?              �?              @      �?              �?      �?      @       @       @       @      �?               @       @      @      @      @      @      �?      @      @      "@      @      @      @      @      &@      (@      "@      .@      (@      4@      ,@      1@      0@      8@     �@@      5@      .@      8@      8@      A@     �@@     �E@     �H@      L@      J@      F@     �F@      G@     �L@     �I@      L@     �I@      A@      A@      B@     �F@      A@      >@      @@      >@     �A@      9@      <@      5@      6@      0@      .@      1@      1@      .@      0@      .@      *@       @      @      &@      @      @      @      @      @      @      @      @       @       @      �?       @      �?              �?      �?              @       @      �?       @              �?        r�.�3      �=%�	�`�|���A*�g

mean squared error��=

	r-squaredx�>
�L
states*�L	   �S��    X�@    �&A!/�'
�P�@)�)�=�H�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @      @      @      @      @@     �R@      R@     �X@     �c@     @n@     �|@     `�@     �@     ڪ@     t�@     ˳@     l�@     ��@     ~�@     ��@     ��@     &�@     �@     ��@     ��@     �@     B�@      �@     ��@     �@     ȩ@     ��@     ��@     
�@     ̰@     N�@     Ĵ@     ��@     B�@     Y�@     ;�@     ��@     U�@     %�@     V�@     V�@     a�@     �@     �@     �@     ��@     ŷ@     ��@     ÷@     ��@     �@     ʺ@     Ļ@     ��@     Q�@     p�@     ��@     ��@     �@     ��@     ��@     �@     b�@     ��@     ��@     Ȧ@     *�@     ڡ@     ؟@     $�@     ��@     D�@     �@     h�@     X�@     Е@     0�@     4�@     ��@     ��@     ��@     ��@     X�@     (�@     0�@     (�@     ��@     ��@     p�@     0�@      �@     �@     x�@     X�@     ��@     X�@     ȃ@     ��@      �@     ��@     @�@     ��@     ��@     ��@     ��@     ��@      �@     p�@     �}@     �~@     P~@     ��@     �z@     �z@     0|@     �}@     p�@     �y@     �x@     y@     `x@     @x@     �v@     �x@     px@      t@     �u@     `u@     @t@     pu@     u@     �s@     �s@     �r@     �q@     �q@     �p@     `p@     q@     `p@     pp@      o@     @n@     `m@     �o@      q@     r@     �q@     `n@     `j@     �i@     �i@     �i@     `g@     `h@     �h@     q@      o@      h@     �f@      h@     �h@     �m@     �i@     �e@     @d@      e@     �a@      d@     �a@      c@     @_@      b@     �`@      a@     �^@      ^@     �[@      ^@      a@     �_@     �`@     �e@     �`@     �X@      ^@     @Y@     @X@     @\@     �[@     �[@     @Z@     @V@     @W@      \@     �U@     �P@     �Y@     @U@     �T@     �T@     @g@     �W@     �R@     �T@     @T@      T@     @Q@      S@     �R@     �Q@      Q@      T@     @S@     @Q@     �J@     �J@     �P@      L@      J@     �J@      G@      K@     �L@      M@      M@      L@      L@      E@      D@     �H@      J@      L@     �L@      F@     �D@      K@     �H@     �F@      H@      E@      G@     �D@      E@      I@     �H@     �D@      D@     �B@      C@      @@      ?@      ?@      ?@      B@      @@      >@      :@      >@      ;@      C@      ;@      B@      <@      B@      3@      :@      <@      >@     �@@      ;@      9@      ?@      8@      0@      5@      7@      ;@      5@      ?@      >@      ;@      8@      7@      9@      =@      2@      &@      4@      3@      5@      1@      3@      3@      *@      *@      0@     �@     p�@      $@      0@      "@       @      0@      (@      1@      &@      ,@      3@      *@      0@      8@      5@      8@      6@      .@      4@      8@      6@      6@      8@      2@      "@      2@      4@      5@      1@      0@      9@      7@      7@      4@      9@      7@      9@      ?@      :@      ;@      <@      8@      A@     �B@     �D@      B@     �C@      >@     �B@      >@      ?@      B@     �B@      C@     �A@      D@     �C@      B@     �G@      D@     �J@      B@      H@     �E@     �F@      F@      P@     �N@      F@     �O@     �J@     �I@      K@      K@      D@      M@     �N@      O@     �E@      G@     �J@      P@     �L@      N@      K@     �P@      M@      S@     �P@     @P@      T@     �Q@      R@     �R@      Q@     �S@      S@     @S@     @R@     �R@      T@     �S@     �T@      V@      X@     @[@     @V@     @W@      V@     @V@     �S@     �Y@      Z@     �Y@     �X@      \@     @]@     @\@     @_@      `@     @_@     �Z@      Y@      b@     �Z@     @_@      a@     �`@      `@     @]@     @[@     �_@      a@     �^@     �`@     �e@      c@      d@     �b@     �c@      e@      d@     `b@      e@     `e@      e@      e@     �f@     �e@     �e@     �j@     `r@     `i@     �h@     `i@     �h@     �l@      j@     `j@     @j@     �k@     �l@     �k@      l@     �n@     �l@      o@     �m@     @m@     Pp@     �p@     �o@     �p@     `q@     `p@     q@     �t@     px@      s@     �r@     �s@     `w@     `t@     �u@     �u@     �s@     �w@      w@     0x@     Px@     0z@     pw@     �z@     P{@     P|@     �y@     H�@     ��@     p�@     ��@     �@     ��@     ��@     ��@     (�@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ��@     ��@     0�@     @�@     @�@     H�@     D�@     ��@     ��@     t�@     �@     ��@     \�@     �@     ��@     X�@     ��@     F�@     ��@     ��@     ~�@     (�@     ��@     �@     P�@     �@     ��@     ��@     ?�@     �@     ϸ@     ]�@     T�@     ͼ@     �@     `�@     W�@     ~�@     �@     >�@     �@     G�@     $�@     ��@     1�@     s�@     ļ@     ��@     ��@     �@     ��@     �@     �@     /�@     8�@     ճ@     ʱ@     �@     ��@     f�@     Ы@     
�@     ~�@     b�@     �@     ��@     �@     �@     ��@     �@     ��@     ܩ@     ¬@     ��@     Բ@     �@     ��@     ߳@     ��@     �@     �@     ��@      @     �|@     (�@     @     @X@      E@      *@      4@      �?        
�
predictions*�	    �ӿ   ��}@     ί@!  �֩jI@)p��D�P@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�>h�'��f�ʜ�7
������6�]������%�>�uE����>8K�ߝ�>�h���`�>>�?�s��>�FF�G ?1��a˲?6�]��?����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@u�rʭ�@�DK��@�������:�              �?      �?       @      @      @      $@      @      "@      1@      2@      ;@      ;@      ?@     �@@      I@     �H@     �N@     �P@     �N@     �M@      Q@     @S@     @T@     �U@     @T@     @V@     �Q@     �S@      M@     �O@      P@      R@      J@     �J@      J@      @@     �L@     �B@     �J@      C@      C@      :@      6@      7@      1@      2@      5@      2@      3@      .@      $@      ,@      .@       @      *@      @      @      @       @      @      @       @      �?      @       @      @       @       @      @      @      @       @       @       @       @              �?      �?              �?       @              �?       @      �?              �?              �?              �?              �?               @              �?      �?       @      �?              �?               @       @      �?      �?      �?      @      �?      �?       @      @      @               @      @       @      @      @      @      @      @      @      @       @      @      @      @      "@      *@      @      &@      (@      ,@      ,@      4@      9@      4@      ;@      =@      @@      :@      @@     �B@      G@      9@      G@     �E@     �I@     �G@      P@      J@     �K@      H@      G@     �K@     �O@      K@      M@      G@      I@     �D@      B@      G@      C@     �A@      6@      B@      <@      >@      =@      6@      @@      ,@      3@      9@      6@      3@      1@      0@       @       @      $@       @      1@       @      @      "@      @      @      @      @      @       @      �?       @       @      �?      �?      �?      �?              �?              �?        =l.��3      +|�	��|���A*�g

mean squared error��
=

	r-squared8��>
�L
states*�L	    ���   ���@    �&A!�Μ�b�@)>;� Ą�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @      @      @      @      *@     �K@     �G@      T@     @[@     �e@      z@     ��@     b�@     L�@     �@     j�@     ��@     b�@     ~�@     ��@     ��@     d�@     أ@     ��@     r�@     ^�@     `�@     :�@     �@     Ԧ@     V�@     ڦ@      �@     6�@     ��@     �@     <�@     ��@     Z�@     ��@     ƻ@     ��@     �@     !�@     ��@     ��@     ��@     7�@     ��@     ��@     �@     -�@     ݶ@     Ͷ@     �@     ��@     ��@     r�@     q�@     �@     �@     G�@     ��@     ʶ@     ´@     ��@     X�@     ��@     "�@     ħ@     ��@     0�@     Ģ@     ��@     D�@     ��@     ��@     �@     ��@     ��@     ��@     ؕ@     l�@     d�@     ��@      �@     x�@     �@     ��@     `�@     p�@     ��@     ��@     ��@     ��@     @�@     ��@     ��@     �@     ��@     8�@     (�@     H�@     x�@      �@     ��@     X�@     ��@     ��@     h�@     x�@     P�@     (�@     @}@     �~@     P|@     �|@     �}@     �{@     �|@     �|@     @}@     �x@     �w@      |@     �x@     �x@     �w@     p{@     �{@     v@     �w@     v@     �t@     pt@     �t@     �t@     �t@     �s@     �r@     0r@      s@      q@     `p@     pq@     0r@      r@     �o@     @p@     pu@      q@     Pq@     @o@      l@     �l@     �l@      m@     �j@     �i@     �h@     �l@     �g@     �h@     `k@     �k@     0p@      j@     �i@      e@     `c@      k@     s@     @f@     �d@     @b@     `c@     �a@      c@     �c@     �a@     @a@     `e@     �g@     @^@     �^@     �_@      \@     �W@      ]@      \@     �T@     �Y@     @\@     �\@     �Z@      Z@     @[@     �W@      X@     �W@     �V@     �V@     @W@      [@     �V@     �V@     @R@     �V@     �U@      U@      W@     `b@      c@     �S@     �U@     �P@      L@     �Q@     �R@      O@     �Q@     �Q@      Q@     @Q@     �Q@     �R@     �P@      P@     �Q@      P@     �P@     �P@      Q@     �L@     @P@     �O@      O@     �D@     �I@     �E@     @P@     �C@      P@      J@      J@      J@      H@      H@      F@     �C@     �D@     �C@      <@      B@      F@      @@      B@      :@      A@     �F@      E@      @@      <@      B@     �C@      9@     �@@     �B@     �B@     �@@      9@      >@      :@      9@     �@@     �@@      4@      8@      >@      :@      ;@      9@      ?@      6@      6@      6@      >@      8@      2@      2@      3@      7@      1@      4@      1@      5@      .@      0@      3@      3@      6@      0@     ��@     �@      .@      (@      &@      &@      ,@      4@      6@      ,@      2@      ,@      5@      2@      1@      ,@      .@      ,@      8@      &@      .@      5@      3@      3@      5@      .@      (@      0@      2@      2@      B@      ;@      9@      =@      @@     �B@      A@      ?@      =@      8@      A@      :@      =@     �A@     �C@      C@      @@      >@      >@      E@     �F@      B@      9@      C@      B@      F@      ?@      >@      G@     �D@      E@      G@      I@      @@      E@     �K@      L@     �E@      K@     �L@     �K@      M@     @Q@     �D@     �H@     �D@      M@      J@     �N@     �J@      Q@     �Q@     �P@      L@      I@     @P@      Q@     @R@      Q@     �T@      K@      U@     �Q@     �G@     �P@      R@     �T@      U@     �U@     �R@     @S@     @T@      X@      V@     �V@     �T@     �S@     �Y@     �W@     �R@     @X@     �Y@      X@      W@     �Y@      ^@     �\@     �^@     �]@     @X@      ]@      ^@      ^@     �`@     ``@     `a@     �^@      ^@      a@     �^@      b@     �a@      `@     @`@     �b@      d@     �`@      a@     �`@      c@     �a@      d@     �c@     �d@     �f@     `e@     �r@     �i@     �f@     @d@     �e@      h@      f@     �h@     `h@     �j@     `i@     �h@     �h@     @l@     �h@     �j@     @m@     `k@      i@      p@     �o@     �n@      n@      m@     @n@     �p@     `p@     �n@     �n@     �q@     `q@     @t@     `{@     @r@     �q@     �s@     �s@     �u@      v@     Pt@      v@     `v@     `w@     @v@     �v@      y@     �u@     �y@     Px@     �~@     `�@     �@     �|@     �|@     P�@     ��@     ��@     `�@     P�@     h�@     ��@     @�@     P�@     ��@     X�@     ��@     P�@     H�@     ؉@     ��@     ��@     ��@     �@     ��@     l�@     �@     ��@     4�@      �@     ��@     ��@     Ț@     h�@     ��@     ��@     ��@     b�@     �@     p�@     Z�@     �@     �@     �@     l�@     |�@     S�@     ¹@     �@     ��@     0�@     g�@     »@     �@     :�@     �@     �@     ��@     ��@     ��@     ��@     C�@     ʺ@     �@     �@     �@     ��@     ڿ@     �@     X�@     %�@     M�@     C�@     K�@     ��@     6�@     f�@     ��@     ��@     f�@      �@     D�@     ��@     \�@     p�@     ��@     h�@     ��@     �@     Ԫ@     ��@     ��@     ��@     y�@     ��@     ��@     ��@     B�@     Њ@      �@     x�@     �{@     �@     ��@     @R@      :@      9@      6@      @        
�
predictions*�	   �;�ο   �=@     ί@!  ���?@)y��D��L@2��Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]�����[���FF�G �>�?�s�����(��澢f���侙ѩ�-�>���%�>�uE����>�f����>��(���>>�?�s��>�FF�G ?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�Š)U	@u�rʭ�@�������:�               @      @      @      @       @      $@      2@      :@     �@@      :@      @@     �E@      A@      J@     �L@     �L@     �S@     @S@     �P@     �S@     �R@      Q@     �T@     �S@     �R@     @S@     @Q@     �P@      K@     �Q@     �K@     �N@     �M@      E@      C@      B@      B@      ;@      9@      :@      9@      =@      =@      1@      ;@      5@      ,@      &@      &@      "@      ,@      $@      &@      (@       @      $@      @      @      @      @       @      @      @      @      @       @       @      �?      �?      @      @      �?      �?      �?              �?       @      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?              �?              �?      �?      �?              �?      �?              �?              �?      �?               @      �?      �?              @      @       @      @      @      @      @      @      @      @      @      @      @      @      &@      @      (@      $@      2@      @      4@      6@      2@      5@      5@      5@      :@     �A@      ?@      D@      E@     �G@     �E@     �F@     �J@      H@      Q@     �Q@      F@     �Q@     �Q@     �N@     �N@     �J@     �I@     �H@      N@     �I@     �C@      B@     �D@     �B@      8@      ?@      5@      ?@      4@      7@      0@      4@      1@      (@      1@      (@      ,@      .@      3@      $@       @      $@       @      @      �?      @      @      @      @      �?      �?      @      @      �?       @      �?      �?              �?      @      �?      �?              �?        ��T4      e(��	!V�|���A*�h

mean squared errormq=

	r-squared��>
�L
states*�L	    ���   `Cu@    �&A!:����D�@)�����@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      @      @      *@     �G@      G@      S@     �\@     �j@     �}@     ��@     غ@     ��@     ��@     �@     
�@     f�@     "�@     <�@     J�@     T�@     ơ@     p�@     �@     ,�@     ��@     &�@     ��@     V�@     �@     D�@     ��@     �@     F�@     ��@     �@     ��@     ̶@     l�@     ¼@     �@     �@     ��@     E�@     �@     Ļ@     �@     d�@     t�@     �@     �@     �@     ��@     �@     �@     �@     ι@     �@     �@     |�@     ^�@     ��@     h�@     N�@     ��@     а@     ҭ@     
�@     �@     F�@     ΢@     ��@     <�@     �@     �@     �@     d�@     `�@     ��@     �@     ̔@     (�@     D�@     ��@     �@     8�@     x�@     ��@     ȋ@     @�@     x�@     X�@     ��@      �@     Ȋ@     p�@     ��@     ��@     ��@     �@     ؅@     P�@     p�@     ��@     ��@     �@     ��@     ��@     Ё@     (�@     �@     P�@     P{@     ~@     �@     �@     �}@     P|@     �~@     �|@     py@      y@     �y@     �x@     �x@     �w@     �w@     pv@     �u@     �v@     �v@     w@      x@     w@     �v@      v@      u@     �s@     @s@     �s@     �s@     Pr@     �s@     Pq@     Pq@     �m@     �p@     �n@     `o@     �n@     �n@     �p@     Pp@      j@      n@     �m@     @r@     �j@     �i@     �g@     `f@     �j@     `o@     �h@     �g@      g@     �d@     `d@      e@     �q@     `k@     �i@     �u@      g@     �f@     �e@     �d@      f@      b@     `c@     �c@     @c@     @_@     ``@     �^@      _@      `@     ``@     @_@     �a@     �`@     �]@      _@      Y@     �\@     �[@     �Z@     @X@      \@      Y@     �Z@      Z@      X@     �X@     @X@     �Z@     @\@     �W@     �Y@     @U@     @V@     �V@     �Y@      V@     @U@      W@     �a@      U@     �Q@     @P@     �N@      T@     �M@     �R@     @Q@     �P@     @Q@     �Q@     �O@      V@      K@      H@     �Q@     �L@     �O@      J@     �Q@      L@     �M@      J@     �C@     �M@      K@      K@      I@     �K@      F@      F@      E@      A@      J@     �D@      I@     �I@     �C@      C@      F@     �@@      G@      A@     �B@     �@@      ;@      E@     �F@     �@@      C@      I@      6@      >@      B@      9@      6@      8@     �A@      A@     �A@      <@      :@      3@      4@      4@      4@      7@      5@      9@      9@      =@     �@@      5@      <@      <@      (@      6@      5@      6@      ,@      &@      5@      2@      7@     ��@     P�@      3@      2@       @      3@      1@      4@      &@      *@      3@      (@      6@      6@      .@      3@      1@      4@      7@      :@      ;@      4@      5@      2@      ?@      3@      ;@      4@      <@      @@      :@      :@      :@      @@      =@      9@      ?@      :@      :@      B@      @@     �C@     �A@      @@      ?@     �@@     �B@      <@     �B@     �B@      C@     �B@      D@     �G@     �H@      H@      >@      E@     �F@     �B@     �G@     �K@     �G@     �H@     �F@      H@      K@     �M@      K@      N@     �J@      G@     �K@      K@     �O@     �J@     �D@      M@     @R@      M@     �K@     �O@     �K@     �Q@     �O@     @P@     �K@     @P@      T@      P@      S@     @W@      S@     �P@     �R@      O@     �V@     @S@     �Q@     �R@     @V@     @U@      T@      T@     �X@     @X@     �Y@     �X@     @[@      W@     @V@     �Z@     @Y@     @^@     @X@     @U@     �W@     �Y@      _@     �Y@     �\@      ]@     �Z@     @\@     �^@     @X@     @_@     �W@     �`@      ]@     �_@     @_@      ^@     `b@     �e@     �a@     @a@      e@     �e@     �c@     �a@     @k@     �m@     @e@     �b@     �c@      d@     �e@     @f@      f@     �f@     @h@      e@     `g@     `f@     �h@      k@      h@      d@     �e@     �j@     @h@     �m@     �k@     �i@     �j@     `j@     �m@      m@      p@      l@      p@     �p@     �o@     @p@     �o@     0q@     �p@     �p@     �|@     �t@     �r@     �q@     @r@      u@     0u@     v@     pu@     �x@     �w@     �w@      w@      }@     �}@     py@     �y@     �{@      @      @     `~@     �|@     �~@     P@     Ȁ@     H�@     (�@     H�@     ��@     (�@     ��@     ȅ@     ��@     ��@     ��@     @�@     �@     Ȍ@     8�@     P�@     �@     ��@     T�@     ��@     0�@     ��@     ĕ@     ė@     ��@     К@     ��@     N�@     ��@     ̣@     ^�@     2�@     ��@     <�@     X�@     j�@     �@     ��@     v�@     ��@     ��@     ˼@      �@     U�@     �@     ^�@     ݸ@     ��@     �@     ��@     "�@     �@     ׷@     ��@     غ@     
�@     n�@     9�@     ��@    ��@    �c�@     о@     G�@     7�@     ʵ@     �@     ��@     h�@     ~�@     6�@     &�@     �@     6�@     Ƨ@     4�@     Ƥ@     ԧ@     ��@     �@     J�@     L�@     V�@     "�@     9�@     B�@     K�@     ��@     )�@    ��@     �@     0�@     ��@     �@     ؀@     p~@      �@     �\@      ?@      ?@      5@      @        
�
predictions*�	   �T�ؿ    ��@     ί@!  �=x,�)I�2a�G@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x����[���FF�G ��ߊ4F��h���`ѩ�-߾E��a�WܾO�ʗ��>>�?�s��>��[�?1��a˲?6�]��?��d�r?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@u�rʭ�@�DK��@�������:�              �?              �?              �?              @      @      @      @       @      $@      4@      :@      ?@     �A@      =@      C@      H@     �F@     �L@      L@     �S@     �U@     �T@      U@     �V@     �W@      X@     �X@     @X@     �T@     �S@     �S@      T@      T@     @S@     �P@     @P@      M@     �J@     �H@     �J@      A@      @@     �E@      5@      7@      @@      5@      8@      4@      1@      ,@      .@      *@      .@      @      @       @      (@      @      @      &@      $@      @      @       @       @      @       @       @      @      �?      @      �?       @      �?      �?      �?       @      �?      �?              �?      �?              �?              �?              �?              �?              �?               @              �?      �?              �?       @              �?       @       @              �?      �?      @               @       @      @      �?      @      �?      @      @      @      @      @      @       @              @       @      "@      (@      *@      @      &@      @      "@      (@      ,@      3@      1@      6@      :@      9@      =@      ;@      7@     �A@     �C@     �B@      I@     �E@      K@      F@      J@      D@     �P@     �L@     �M@     �K@     �D@     �I@      ?@      C@     �A@      9@     �A@      6@      :@      =@      <@      0@      ,@      5@      <@      "@       @      *@      (@       @      *@      "@      "@       @      @      @      "@      @      @      @      �?      @      @               @       @       @              @      �?      @      �?      �?       @      �?      �?              �?              �?              �?        ���R3      ��	#�	}���A*�f

mean squared error��=

	r-squaredL�>
�L
states*�L	   @���   �|x@    �&A![pZ�y�@)���z�i�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @      @       @      (@      8@      M@     �S@      \@      e@     Ps@     �@     ��@     ��@     �@     p�@     ��@     :�@     �@     ��@     ��@     ��@     ��@     �@     J�@     �@     &�@     �@     ��@     ��@     x�@     ��@     D�@     d�@     ��@     |�@     Y�@     �@     �@     �@     �@     d�@     M�@     ľ@     ��@     E�@     Y�@     Ի@     j�@     ��@     ;�@     W�@     ڶ@     �@     p�@     �@     K�@     �@     ĸ@     ҹ@     �@     V�@     q�@     ��@     �@     ��@     Ҳ@     z�@     0�@     ,�@     Ψ@     "�@     ��@     �@     ޠ@     �@     ��@     �@     t�@     ��@     ��@     �@     ̓@     �@     ��@     ,�@      �@     ��@     8�@     ��@      �@     �@     ��@     `�@      �@     ��@     (�@     ��@      �@     ��@     ��@     P�@     0�@      �@     ��@      �@      �@     `�@     �@     @�@     ��@     ��@     0@     0@      ~@     p}@     �@     `}@     �{@     `}@      |@     pz@     �{@     @|@      x@     �x@     0v@     pw@     0u@     `y@     u@     �u@     �t@     �u@     u@     @r@     �s@     r@     `r@     �r@     0r@     �u@     v@     �q@     pr@      r@     �q@     �n@     `q@     `o@     `q@      q@      o@     `m@     @o@     `n@      i@     `j@     �i@     @i@     �j@      k@     �i@     @h@     �j@      v@     @r@     pq@     �i@     �o@     �f@      b@     �e@     �d@     �e@     �a@     �a@     @d@     @a@     �c@     @b@     �b@      f@     @k@     `e@     �\@     �^@      _@     �`@     `c@     �a@     �_@     `b@      a@     �^@     @[@     �\@     @W@     @_@     �W@      \@     �[@     �]@     �U@     �W@      W@      W@     @U@     �U@     @V@     �]@     @Y@      S@     �T@     @V@      V@     @U@     �R@     @R@     �P@      R@      V@     �X@     �`@     @R@      L@      S@     �Q@     �O@     �Q@     �R@     �Q@      S@     �P@     �O@      O@     �N@     �J@     @P@      I@     @P@      H@      O@      M@      M@     @P@     �I@      G@      K@      I@      B@     �E@     �O@     �D@      G@      =@     �H@      G@     �C@      F@     �E@      >@     �@@     �@@      C@      D@     �B@      =@      6@      C@      E@      6@      7@      ;@     �A@      9@      :@      7@     �B@      ?@      ;@      <@      5@      :@      4@     �@@      5@      4@      8@      1@      .@      8@      6@      2@      3@      0@      0@      2@      2@      1@      3@      6@      8@     Ѝ@     �@      .@      1@      $@      2@      9@      0@      4@      1@      8@      3@      6@      3@      8@      6@      7@      (@      7@      9@      <@      7@      3@      5@      :@      ,@      ?@      =@      6@      @@      9@      6@      @@      5@      9@      >@     �@@     �D@      <@     �D@     �B@      C@     �E@      ;@     �D@      8@     �F@     �B@     �E@     �B@     �G@      ?@     �B@      C@      J@     �H@     �B@      C@     �B@      E@      J@      F@      I@      I@      B@     �H@     �H@     �J@     �C@     �H@     �O@      I@      F@     �O@      H@     �N@     �M@      P@     �M@      P@     �P@      P@     �P@     �Q@     @S@     �P@     @U@     �P@     �M@     �M@      L@     �Q@     �Q@     �Q@     �S@      P@     �P@     �Y@     @Q@     @R@      T@     �U@      R@     �Z@     @U@     �S@     �U@     �V@     @T@     �X@     @U@      Z@      X@     �[@      X@     �\@      Y@      ]@     @Z@      X@     @]@     �[@     �^@      _@      ]@     @\@      ]@     �^@      ^@     �_@     @`@      ^@     @\@     �`@      _@     �c@     `a@     @c@      a@      l@     �m@     �b@     �b@     �d@     @c@     �c@      c@     @d@     �e@     �b@     @g@     @d@      g@      e@      g@     �h@      f@     `i@     �i@     �l@     �f@      h@     �i@     �i@      j@      n@     �n@     `k@     �n@     �j@      n@     `n@     0p@     �o@     `n@     �p@     @r@     �k@     Pp@     @q@     �y@      u@     Ps@     0s@     �s@     �s@      w@     �w@     �u@     �w@     �v@     `�@      x@     �y@     y@     @{@     �z@     �}@     0@     H�@     ��@     ��@     �@     p@     ��@     ��@     ��@     ��@     ؅@     ��@     P�@     8�@     ��@     ��@     ��@     �@     0�@     ��@     T�@     ��@     �@     |�@     ��@     x�@     D�@      �@     X�@     h�@     p�@     `�@     0�@     2�@     ��@     ��@     T�@     |�@     d�@     ��@     
�@     ��@     P�@     ,�@     ��@     6�@     �@     �@     Z�@     s�@     N�@     ��@     E�@     �@     r�@     ݶ@     ¸@     �@     ��@     ˻@     ��@     U�@     ��@     6�@     �@     ��@     �@     ��@     ;�@     Ӵ@     �@     ��@     ܮ@     &�@     d�@     �@     �@     (�@      �@     4�@     إ@     ��@     ֧@     ħ@     ީ@     �@     ��@     ��@     �@     ��@     ��@     ��@     �@     ��@     ��@     �@     �@     Ѐ@     ��@     ��@     ȁ@     @X@     �B@      >@      ;@      @        
�
predictions*�	    ��ڿ    ��@     ί@!  ��K@)�MQ���Q@2�W�i�bۿ�^��h�ؿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
������6�]���6�]��?����?x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�DK��@{2�.��@�������:�              �?              �?      @      @       @      @      .@      $@      *@      :@      >@      =@     �B@      F@     �I@      I@      L@      T@     �W@     @S@     @V@     �T@     �Q@     �T@      U@     �S@     �Q@     @S@      Q@     �Q@      Q@     �N@      G@     �J@     �H@      F@      D@     �D@     �@@      B@     �@@      A@      (@      3@      ,@      &@      2@      &@      3@      0@      @      &@      @      &@      $@       @      @      (@      "@      @      @      @      @      @       @               @               @      @      @      �?       @               @       @      @      �?       @      �?       @       @       @              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?      �?               @              �?       @      @              @      @      @      @      @       @      @      "@      @      $@      @      $@      @      "@      1@      (@      *@      *@      "@      :@      5@      C@      =@      >@      ;@      ?@      A@     �@@      C@      E@     �B@      E@      F@     �K@     �J@      N@     �O@      L@     �K@     �N@      H@      L@      F@     �I@      A@     �D@     �F@      G@      B@      >@      9@      B@      ;@      3@      ?@      :@      7@      =@      3@      7@      4@      3@      $@      0@      @      @      @      @      $@      @      @      @      @       @      @       @              �?               @       @              �?      �?      �?      �?              �?        ڏ�ٲ3      _  	`:}���A*�g

mean squared error��=

	r-squaredy�>
�L
states*�L	   @���    !�@    �&A!��	\P�@)��^���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      @      @     �@@      R@      U@      _@     @j@      z@     X�@     ��@     ��@     ة@     X�@     =�@     �@     p�@     Ҧ@     ��@     p�@     f�@      �@     B�@     �@     ̢@     ,�@     f�@     ��@     
�@     ڤ@     ��@     ��@     f�@     t�@     >�@     ��@     P�@     ��@     M�@     ػ@     ��@    �0�@     ��@     �@     ��@     �@     ��@     Թ@     ��@     �@     #�@     ��@     ��@     �@     |�@     ѹ@     c�@     �@     ѻ@     �@     ��@     8�@     �@     
�@     ��@     ��@     Ԭ@     X�@     ��@     ��@     ܢ@     ��@     �@     ��@     P�@     ؙ@     ��@     ��@     d�@     ��@     ��@     0�@     ��@     ��@     �@     �@     Ȏ@      �@     ��@     �@     ��@     ��@     ؆@     ��@     ��@     ��@     �@     �@     ��@     ؁@     ��@     ��@     p�@     0�@     ��@     �@     ��@     ��@     ��@     p@     0~@     @@     0�@     ��@     P{@     P}@     `}@     �z@     �{@     `x@     �x@     �{@     �y@     `x@     @u@     �y@     �u@     0v@     pu@     pu@     �v@     Pt@     s@     Ps@     @s@     �r@     0q@     Pq@     Pt@     0r@     �p@     pp@     �r@     �n@      p@     �s@     �o@     �n@     �n@     �o@     �m@     `p@      n@      l@      l@     �i@     �j@      m@     �t@      i@     �k@     �j@      k@     �f@      l@     �f@      h@      f@     @f@     �d@     �e@     0r@     `b@      e@     �d@     �e@     `e@     �a@     `c@     �c@     �a@     �c@     `b@      a@     �b@     @`@     ``@     �d@      n@      `@     �_@     �^@     �^@     �]@     �`@     �W@     �[@     �W@     �`@     �X@     @Z@     @W@      W@      V@     �[@     @V@      X@     �U@     �W@     �W@     @Y@      S@     �S@     �U@     �S@     @P@     �R@     @T@     �P@     �T@     @V@     �W@     @S@     �W@     �U@     �_@     �T@      P@     �M@     @Q@      M@      R@      O@     �R@      L@      H@      H@      C@     �M@      G@     �Q@     �F@     �I@      M@      G@      I@     �L@     �J@     �O@     �F@      D@      I@     �N@     �E@      G@     �@@      F@     �E@      ;@     �A@      <@      ?@      B@     �D@      8@      G@      >@      D@     �@@      A@      ?@      6@      <@      =@      <@      ?@      :@      5@      =@      @@      =@      A@      >@      ;@      5@      :@      ?@      7@      9@      6@      ,@      2@      9@      8@      0@      .@      1@      :@      ,@      5@      ,@     ؎@     (�@      2@      3@      4@      1@      7@      6@      4@      5@      3@      6@      <@      8@      ;@      5@      6@      8@      7@      8@      :@      ;@      8@      >@      9@     �C@      8@      :@      :@      ;@      ?@      ;@      7@      <@      ?@      @@      >@      @@      :@      D@      A@      D@      E@      E@      >@     �E@     �G@      >@      E@     �C@      A@     �B@      G@     �C@     �D@     �E@      C@     �K@      H@     �B@     �G@     �L@     �J@     �L@     �I@      H@     �L@      I@      M@     �K@      F@     �P@      O@     �K@     �I@     �P@     �Q@     �G@      P@     �F@     �Q@      Q@      M@     �N@      O@     �S@     �M@      T@      N@     �S@     �Q@     �M@      V@     �S@     @R@      W@     �S@     �U@     @V@     @R@     �T@     �T@     @Q@     @W@      V@      T@      V@     �V@     �W@     @S@      [@     �X@     �X@     �Z@     �Z@      V@     �Y@      X@     �^@     �\@      \@     �\@      [@     `a@     �`@     �Y@      ]@     �]@      `@     @`@      ]@     �]@      c@     @`@      c@     �a@     �a@     �d@     �p@     �c@     �d@     �d@     �b@     �d@      e@     �c@     �e@     �f@     �d@      e@     �f@     `f@     �d@     �i@     �g@     @g@      i@      h@      h@     �g@     �i@     �h@      h@     �m@      j@     �k@      k@      m@     �j@     �n@     �o@     `p@     �o@     @o@     �m@     �p@     �p@      q@     �q@     Ps@     `s@     `p@     �t@     �t@     Ѐ@     �w@     �u@      v@     pv@     0}@     0y@     �w@     Pw@     �x@     �y@     @z@     �y@     `}@      |@     �}@     �~@      @     ��@     Ȃ@     �@     ��@     ��@     h�@     ��@     ؄@     ��@     ��@     (�@     ��@     (�@     �@     H�@     4�@     X�@     �@     ��@     @�@     l�@     D�@     Е@      �@     �@     ��@     ��@     ؞@     6�@     �@     ��@     ܥ@     0�@     \�@     �@     ��@     O�@     �@     ʴ@     4�@     "�@     ��@     ѻ@     d�@     ʻ@     ��@     9�@     �@     `�@     ݸ@     ��@     A�@     -�@     ��@     ��@     K�@     Ҽ@     -�@    ��@     ��@     ��@     ��@     z�@     �@     j�@     �@     4�@     ��@     �@     0�@     ��@     2�@     j�@     ��@     .�@     P�@     Ȧ@     ��@     �@     h�@     *�@     ��@     ��@     �@     �@     ��@     L�@     ��@     ϴ@    �?�@     `�@     (�@     `�@      �@      �@     �@     ؀@      c@      D@      >@      A@      @        
�
predictions*�	   �=
ۿ   ��\
@     ί@!  이a8�)����@�I@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�6�]���1��a˲��FF�G �>�?�s���})�l a��ߊ4F����Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@�Š)U	@u�rʭ�@�������:�              �?              �?       @               @      @      @      "@      0@      0@      8@      8@     �A@     �@@      H@     �H@     �M@     @S@     �V@     �R@      [@      U@      X@      `@      \@     @W@     @V@     �V@     �X@     @Z@     @V@     �S@     �T@     �S@     �T@     �P@     �J@     �N@     �R@     �B@     �M@     �C@      C@      A@     �F@      <@      ;@      3@      9@      5@      .@      0@       @      "@       @      ,@      "@      $@      @      @      "@      @      @      @      @      @      @      @      @      @      @       @       @      @      �?              �?       @       @      �?      �?       @               @              �?              �?              �?              �?              �?      �?               @      �?      �?               @      �?              �?              �?      @       @       @              �?       @      @       @       @      �?      @      @      @              @      @      @      @      &@      *@      @      "@      @      (@       @       @      *@      ,@      0@      *@      (@      3@      7@      1@      :@      :@      <@      =@      @@     �C@      ?@      A@      B@      @@      D@      @@      ;@      =@      >@      9@      9@      @@      =@      B@      @@      5@      A@      3@      7@      ,@      6@      2@      1@      3@      4@      3@      ,@      2@      .@       @      "@      "@      $@      "@      @      @       @      @      @      @       @      @      @      �?      �?              @       @       @      �?              �?      �?      �?              �?              �?        �4���3      +|�	B�+}���A*�g

mean squared error��<

	r-squared`�>
�L
states*�L	    ���    "�@    �&A!l�`)8�@)V�s�^��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      @      @      3@     �G@     �U@      Z@     �c@     Pt@     X�@      �@     ,�@     (�@     �@     Ȱ@     ̫@     .�@     ��@     T�@     f�@     �@     ��@     Σ@     F�@     ��@     �@     Ȥ@     ��@     ��@     �@     ��@     ��@     >�@     �@     Ӱ@     ��@     ��@     �@     ܷ@     ��@     ��@     ��@     ��@     ��@     ׼@     {�@     2�@     ��@     ��@     �@     ��@     �@     ׷@     �@     �@     ��@     ��@     e�@     �@     u�@     ��@     ��@     }�@     ��@     _�@     i�@     -�@     ��@     ��@     X�@     ¦@     .�@     �@     ��@     P�@     ��@     �@     �@     ��@     L�@     `�@     ��@     l�@     �@     �@     $�@     ��@     h�@     ��@      �@     ��@     X�@     ��@     ��@     (�@     x�@     X�@     ��@     @�@     ��@     ��@     P�@     h�@     ��@     x�@     ؁@     X�@     ��@     Ѐ@     0@     ��@     �~@     0}@     0@     0~@     �{@     �@     (�@     �y@     �y@     �|@     �x@     0x@     pu@     Px@     �t@     �u@      v@     u@     @t@      u@      r@     0t@     �s@     �s@     0s@     �r@     �q@     �o@     `r@     �q@      q@      o@     `q@     p@     �n@     `o@     �q@     �p@      o@      m@     `m@      l@     @l@     �k@     �m@     �n@     u@     �k@     �k@     `i@     �i@     �f@     �g@     �f@     `h@     �d@     �f@     �i@      d@     �e@     `e@     @c@     �d@     �b@     `e@      r@     @f@     �c@      e@     �b@     �`@     @a@      _@      Z@     �`@     @Z@     @^@     @]@     �^@      ]@      a@     �\@     @n@      `@     �\@     �Z@     �]@     �[@      W@      Z@     @[@     �^@      \@     �U@     �X@      \@      Z@     �V@     �S@     �W@      V@     �W@     �V@     @U@     @R@      S@     �Q@      W@      R@     �P@     �S@     �U@     �V@     @V@     �T@     �Q@      ]@     �X@      S@      R@     �M@      Q@      P@      R@     �N@      P@     �K@      L@      G@      M@     �N@     �M@     �I@      K@     �D@      H@     �K@      E@     �K@     �K@      E@      E@     �I@     �I@     �A@     �A@      :@      ;@      G@     �@@     �B@      D@      E@      C@     �B@      ?@     �B@      A@      B@      =@     �E@      C@      9@     �B@      <@      :@     �B@      7@      9@     �@@      <@      =@      7@     �B@      ;@      >@      3@      :@      3@      6@      3@      7@      7@      4@      3@      7@      *@      6@      1@      $@     P�@     ܑ@      1@      :@      ;@      5@      0@      2@      :@      0@     �@@      9@      6@      3@      7@      :@      1@      2@      6@      4@      5@      ;@      :@      >@      7@     �@@     �@@     �@@      8@      ;@      A@      >@      <@      ;@      A@      ?@      A@     �A@      E@     �E@     �G@     �@@      =@      B@      A@      B@      B@      F@     �C@     �B@     �C@     �C@     �B@      D@      I@     �C@     �A@     �G@      C@     �H@     �I@      L@     �K@      K@     �D@      H@      Q@      C@     �I@     �N@     �G@      K@     �J@     @Q@     �J@     �O@     @P@     �M@     �O@     @S@     �N@      R@     @P@     �P@     @S@      M@      O@     @Q@     @T@     �N@     �S@     �U@      U@      Q@      W@     �S@     @S@     �S@     �S@     �U@     @T@     �X@     �Z@      V@     �T@      V@     �Y@     @U@     @T@     �V@      U@     �W@     @Y@     �W@      U@     @W@     �V@     �]@      Z@     @`@     �^@     @`@     �Z@     @b@      \@      `@     �`@     �a@      `@      a@     @e@     @`@      c@     �c@      c@     �`@     p@     �a@      d@     `c@     �b@     �d@     �d@     @e@     `c@     `c@     �d@     �d@      h@     `f@     �e@     �f@      e@      f@     @f@     �g@     `j@     `l@     �l@     �g@      k@     �k@      i@     �i@      n@     `o@     �m@     �n@      j@     `o@     �q@     `n@     0p@     pt@      o@     �r@     �q@      q@     s@     �r@     pu@     @t@     �s@     �w@     `v@     �t@     pv@     �|@     ��@     @x@     �w@      {@     @y@     py@     `{@     z@     }@     �|@      }@     @~@     �@     p�@     ��@     (�@     �@     ��@     x�@     @�@     P�@     ��@     ��@     x�@     x�@     ��@     `�@     X�@     �@     p�@     ԑ@     @�@     ��@     ��@      �@     Ĕ@     ��@     �@     ��@     ��@     �@     �@     ��@     �@     �@     �@     &�@     ��@     ��@     ��@     �@     ˳@     8�@     c�@     ĸ@     g�@     �@     ��@     �@     1�@     �@     F�@     s�@     #�@     �@     f�@     Ϸ@     ��@     �@     !�@     \�@     ~�@     ��@     ��@     ��@     )�@     ͻ@     4�@     ��@     i�@     �@     �@     خ@     ��@     2�@     �@     J�@     ި@     .�@     ��@     ��@     ��@     f�@     V�@     �@     ި@     v�@     ��@     ^�@     _�@     ��@     a�@     �@    ���@     ȝ@     ��@     ��@     X�@     �}@     8�@     �@      m@     @R@      C@     �A@      @        
�
predictions*�	    �ݿ   ���@     ί@!  lI�C@)uC��AQ@2���Z%�޿W�i�bۿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9����[���FF�G �>�?�s���O�ʗ����f�����uE���⾮��%���>M|K�>�_�T�l�>E��a�W�>�ѩ�-�>I��P=�>��Zr[v�>1��a˲?6�]��?>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@u�rʭ�@�DK��@�������:�              �?              �?      @      @      @      @      .@      *@      *@      5@      7@      5@      F@     �C@     �K@      I@      J@     �S@     @P@     @S@      R@     @W@      U@      R@     @S@      P@      O@      S@      V@      T@      R@      O@      O@     �N@     �F@     �F@      E@      ;@     �B@     �A@      D@      =@     �B@      3@      9@      ,@      *@      0@      6@       @      ,@      &@      (@      "@      (@      @      $@      @      $@       @      @       @      @      @      @       @       @      �?      �?      �?      �?       @      �?       @       @       @      �?      �?       @       @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @      �?              @      @       @      �?       @              �?       @      @       @      �?      @      @      @      @      @       @      @       @      @      @      @      0@      @       @      $@      1@       @      &@      8@      (@      ;@      6@      2@      7@      3@      <@     �B@      @@      I@      H@     �J@     �C@     �G@     �K@      L@      C@     �G@     �M@      I@      I@     �G@     �G@     �A@      H@     �G@     �I@      J@      E@     �B@      ;@      C@      @@      ?@      :@      >@      4@      3@      1@      1@      0@      4@      7@      1@      &@      ,@      .@      0@      &@      @      @      @      @      @      @      @       @      @       @      @      @       @              �?      @              �?              �?        � �b4      8�W	Mg<}���A *�h

mean squared error?� =

	r-squaredw�>
�L
states*�L	   ����   @�@    �&A! �Dd�w�@)�+����@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      "@      &@      ,@      C@      N@     @T@      e@     0x@     �{@     ��@     Q�@     �@     |�@     3�@     ,�@     X�@     
�@     Τ@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ʣ@     ��@     V�@     ��@     Ȫ@     D�@     ��@     7�@     �@     ��@     7�@     q�@     ټ@     }�@     ��@     ̾@     ߾@     ��@     L�@     �@     ~�@     ��@     m�@     ��@     W�@     X�@     '�@     �@     *�@     �@     `�@     V�@     ޹@     ָ@     ��@     ��@     !�@     �@     ��@     ��@     ��@     �@     �@     ��@     �@     <�@     �@     �@     D�@     0�@     �@     ԓ@     l�@     ��@     X�@     ��@     ��@     ��@     ��@     h�@     H�@     ȉ@     0�@     p�@     X�@     Ѕ@     ��@     @�@     ��@     Ȅ@     x�@     �@     (�@     �@     (�@     ��@     0�@     ��@     �~@     �@     p@     �}@     p}@     {@     �{@     �y@     P}@     �~@     @{@     `x@     �w@     �y@     �z@      y@      v@     �u@     �y@     Pt@      t@     pt@     u@     �s@     �s@      t@     0u@     0q@     r@      r@     @r@     �r@     0s@     �p@     `m@      n@     �l@     `m@     `n@     0q@     `r@     @j@     �l@     0p@     �o@     �w@     Pp@     @l@     �l@      j@      l@      h@      f@     �k@      j@     �h@     �e@     �i@     �f@     �h@     @j@     �f@      g@     �f@      g@     @b@     �a@     �c@     �`@     @b@      b@     �a@     �c@      a@     @p@     @c@     @^@      ]@     @b@     �a@      a@     �]@     �Y@     @\@     @\@      X@     �_@     �X@     @Z@     @[@     �`@     @k@     �b@     @V@     @[@     �W@      \@     �W@      \@     �[@     �Z@      W@     �T@      T@      U@      S@     �P@      U@     �W@     �W@     �T@     �T@      Q@      U@     @U@     �S@     �M@     �Q@     @R@     @R@     �S@     �O@      R@      P@      X@      Z@     �P@     �P@     �S@      M@      P@      M@     �P@     �I@     @P@      G@      Q@      N@     �E@      E@      J@     �E@      J@      H@     �A@     �E@      G@     �H@      C@      E@     �I@     �E@     �C@      F@      A@      H@      E@     �H@      D@     �C@     �@@      H@     �A@      3@      7@      8@      ;@      A@      ;@      ;@      A@      <@      7@      9@      ?@      8@      C@      6@      <@      ;@      8@      4@      3@      5@      3@      4@      9@      7@      9@      .@      7@      .@      2@      1@      0@     8�@     ̒@      ,@      6@      4@      2@      .@      4@      1@      ,@      5@      3@      =@      1@      4@      3@      9@      4@      :@      ?@      8@      6@      >@      9@     �A@      =@      3@      >@      E@      9@      E@      B@      @@     �E@     �A@      D@     �D@     �B@      >@      G@      8@     �A@      @@     �B@      @@     �@@      D@      F@      :@     �A@      A@     �D@      E@     �B@      F@     �B@     �B@     �G@     �J@     �C@      E@      H@      H@     �I@      I@      J@     �I@     �E@      I@     �G@     �K@     �G@     �P@      P@     �K@     �I@     �P@      P@      M@     �Q@     �O@     �R@      R@      L@     @P@     �U@     �Q@      Q@     �M@     �R@     �T@     �R@     @V@     �N@     �T@     �P@     �T@     �Q@     �V@     �U@     �T@     �X@     �U@     �Y@     @S@     @V@     �V@     �Y@     @X@     �Z@     �U@     �U@     �Y@     @W@     �]@     �W@      U@     �\@      X@     �Z@     �^@     @Y@     �[@      \@     �Z@     �[@     �`@     �a@     @X@      c@     �a@     �`@      a@     �c@     �p@     �i@     �b@     �a@      b@     `b@      b@     @d@      d@      e@     �d@     �e@     �d@     `d@     `d@      e@     `d@     `d@     �h@     �f@     �l@      h@     @g@     �j@     �k@      k@     �l@     �j@     �j@     @l@     �j@     `o@      n@      n@     �p@     @q@     �o@     @p@     �q@     �q@      q@     �s@     t@     `u@     pt@     0t@     `u@     @v@      u@     �y@     �u@     �v@     �z@     p�@     Pw@     �v@     �v@     �w@     �x@     `{@     �@     p{@     `~@     H�@     �@     P}@     ��@     0�@     ��@     �@     8�@     ��@     ��@     ��@     �@     ��@     p�@     p�@     ��@     (�@     (�@     `�@     �@     8�@     0�@     ��@     4�@     |�@     �@      �@     ��@     �@     L�@     �@     ��@     r�@     v�@     n�@     Z�@     ��@     B�@     �@     ®@     ��@     ��@     H�@     A�@     6�@     Y�@     ��@     |�@     ��@     ��@     ��@     ��@     ̷@     g�@     '�@     ��@     ķ@     ķ@     Ը@     ��@     ��@     U�@    ��@     d�@     f�@     3�@     ��@     ��@     ��@     �@     �@     |�@     ̰@     d�@     :�@     ��@     &�@      �@     F�@     ܧ@     ��@     |�@     ��@     J�@     P�@     ��@     ��@     �@     �@     ��@     $�@     l�@     ��@     �@     �@     ��@     �@     ��@     (�@     ��@     8�@     �@     �i@     �_@     �B@      >@      @        
�
predictions*�	   �d�ٿ   ���@     ί@!  ԍH3�)��̀<:N@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r���[���FF�G �>�?�s���O�ʗ���I��P=��pz�w�7�����%ᾙѩ�-߾��~]�[Ӿjqs&\�Ѿ�ѩ�-�>���%�>��(���>a�Ϭ(�>})�l a�>pz�w�7�>����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�vV�R9?��ڋ?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@�DK��@{2�.��@�������:�              �?      �?              �?      @      @      @      �?      &@      *@      8@      4@      @@      9@     �G@     �H@      H@     �G@     �I@      T@      S@     �T@      T@      V@      Z@     @X@     @V@     �[@     @W@     �V@     @T@      S@     �V@     @S@     @R@     �N@     �K@     �M@     �K@      K@     �H@      I@      E@      >@      9@      7@      :@      4@      :@      ,@      <@      0@      (@      $@      *@      $@      "@      $@      (@      @      "@      @      @       @       @      @      @       @       @      @              @       @      �?      �?      �?       @      �?      �?              �?               @       @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?               @               @              �?       @      @              �?               @       @               @       @      @       @      �?      @      @      "@      @      @       @      "@      @      "@      &@       @      &@      .@      &@      ,@      .@      6@      1@      1@     �A@      1@      >@      ;@      C@      G@     �B@      F@     �C@      I@     �H@     �K@      G@     �E@     �D@      E@      D@      @@      @@      ?@     �A@      A@      A@      7@      8@      2@      >@      7@      5@      2@      4@      A@      5@      (@      6@      $@      .@      &@      *@      ,@      @      @      @      @       @       @      @      @      @              @      @       @      @      @      �?              �?              �?              �?       @      �?              �?              �?        S�-�r4      d��H	t%M}���A!*�h

mean squared errorz|�<

	r-squaredZ��>
�L
states*�L	   `���   ���@    �&A!V%"n��@)�O!r@��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      $@      &@      4@      @@     �R@     @R@     �a@     0w@     }@     ��@     ��@     �@     �@     ��@     ̫@     §@     ��@     ��@     �@     �@     ��@     ��@     &�@     ��@     B�@     ʤ@     ��@      �@     :�@     ��@     ��@     �@     ��@     N�@     n�@     f�@     ��@     ��@     {�@     �@     ��@     ��@     ��@     ݽ@     v�@     K�@     �@     �@     #�@     �@     ��@     ��@     ��@     Ǵ@     ��@     �@     s�@     ��@     g�@     ��@     �@     Ϸ@     ��@     +�@     ��@     i�@     ��@     `�@     &�@     f�@     ��@     ��@     4�@     ��@     ��@     p�@     (�@     L�@     �@     ��@     |�@     ��@     ��@     ��@     @�@     ��@     H�@     P�@     ��@      �@     8�@     �@     p�@     ��@      �@     h�@     Ђ@     ��@     ��@     (�@     ��@     ��@     �@     x�@     p~@     �@     �}@     �~@     �|@     ��@     Ȃ@     �{@     �y@     �z@     �z@     pz@      x@     0v@     u@      x@     �s@     �v@     @x@     0t@     �u@     �s@     v@     ps@     ps@     �r@     r@     Pq@     q@     �q@     Pq@     �p@     �r@     �n@     �o@     �p@     �p@     �p@     Pq@      p@     �p@     �o@     �m@      p@     `o@     `k@      r@     �n@      j@     �j@     `k@     �j@      j@     @i@     �j@     �h@      j@      i@     @t@     `j@     �f@      i@     @e@     �e@     �e@     �a@     `e@     @g@      c@     ``@      f@     �`@     �`@     �a@     `c@     �`@     �d@     �b@     �p@     `b@      `@     �\@     `a@     @[@     �`@     @\@     �_@     �\@     @[@     �`@     �]@     �\@     �Y@      [@     @_@     �n@      ^@     @V@     �V@     �[@     �^@     �[@      Y@     �U@      X@     @W@      W@     @S@     �Q@     @Z@     @U@      T@      U@      Y@      S@     �R@     �V@     @S@      N@     �W@      S@      M@      Q@      Q@     @T@     @V@     �Z@     �R@     �T@     �K@     @P@     �P@     �Q@      P@     �R@     �S@     �J@      T@     �D@      J@      K@     �P@     �L@      H@      M@      M@      E@      B@      K@      C@      D@     �C@     �I@     �C@      I@     �E@      F@      8@      E@     �@@      B@      E@      =@     �G@      G@      @@     �B@      >@      <@     �B@      C@     �B@      A@      D@      >@      ;@      B@      3@      8@      9@     �A@      <@      @@      ;@      =@      6@      2@      8@      =@      4@      >@      5@      4@      ;@      7@      .@     \�@     (�@      ,@      2@      5@      9@      &@      5@      <@      &@      2@      7@      ;@      5@      5@      8@      5@      ;@      :@      <@      8@      1@      5@      >@      ;@      <@      D@      <@      @@      9@      A@      ?@      C@      ?@     �D@      D@     �E@      @@      =@      D@      ?@     �@@      D@      =@      >@      B@      E@      E@      B@     �D@      E@      B@      D@      F@      E@      G@      D@      M@      J@      H@      F@      E@     �H@     �J@     �J@      D@     �N@      N@     �F@     �P@      M@     �L@      N@      M@      K@      H@      N@     �P@      Q@      R@     �M@      R@      N@     �Q@      R@     �L@     �O@     @S@      U@     �T@     �U@     �R@     �T@     @S@     @W@     �V@     �U@      U@     �R@     @U@     @V@      V@     �X@     �V@     @V@     �U@     �T@     �W@      U@      Z@     @W@     �Z@      [@      U@     �[@     @]@      [@     �Y@      `@     ``@     @b@      Z@     �X@      _@      a@     �\@      a@     @`@      `@      `@     `b@     �a@     �`@     `b@     �p@     �b@     �g@     @c@     �b@     �b@     @e@     @c@     �e@     �d@      d@     `e@      e@     �d@     `e@     @d@     �f@      i@     �g@     `g@     `j@      k@      i@     �i@     �l@     �j@     @l@     �m@      j@      m@     `i@     @l@     p@     �n@     �l@     @n@     @p@     0r@     �p@     �q@     @s@      t@     �v@     Pv@     Pw@     0u@     `t@     �t@     0u@     �t@     w@     �w@     �{@      |@     �w@      z@     �y@     0z@      }@     {@     �{@     ��@     ��@     h�@     ��@     �@      @     ��@     x�@     Ё@     ȃ@     ��@     H�@     ��@     ��@     ��@     8�@     x�@      �@     0�@     0�@     ��@     @�@     h�@     X�@     ��@     ��@     ��@     ��@     Ж@     ȗ@     ��@     x�@     ��@     �@     ��@     ��@     ��@     Z�@     �@     ��@     x�@     ��@     ��@     d�@     ��@     X�@     ��@     0�@     ��@     ��@     F�@     '�@     i�@     y�@     "�@     �@     ��@     ��@     �@     ·@     ��@     A�@     ��@     ��@     '�@    �>�@     k�@     v�@     H�@     �@     �@     p�@     ��@     e�@     i�@     ��@     �@     j�@     r�@     ��@     ��@     �@     ��@     @�@     ��@     ��@     ��@     |�@     ��@     ��@     ҭ@     Ӱ@     .�@     g�@     ,�@     ϵ@    �e�@     ��@     @�@     ��@      �@     ��@      �@     ��@     �l@     �Z@     �E@      <@       @        
�
predictions*�	    �;ڿ   `c~@     ί@!  �D�@@)��'p��P@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��>�?�s���O�ʗ����f�����uE���⾟MZ��K�>��|�~�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@w`<f@�6v��@u�rʭ�@�DK��@�������:�              �?      �?      �?      @      @      "@       @      @      (@      @      3@      2@      7@      =@      =@      ;@      @@      E@      K@      I@     �Q@     @R@      N@     @Q@     �T@     �X@     @R@      Q@      P@      N@     �M@     �Q@      Q@     �P@     �I@     �P@     �O@      N@     �L@     �H@      C@      >@      A@      A@      @@      ?@      5@      ;@      3@      .@      ,@      ,@      2@      "@      &@      $@      @      "@      @      @      @      @      "@      @      @      @      @      @              �?       @       @      @       @       @      �?      �?      �?              @              �?       @              @      �?              �?      �?              �?              �?              �?              �?              �?              �?               @       @               @              �?               @              �?      �?      �?      �?      @              �?       @       @      @      �?      @      @      �?      @      @       @      @      @      @      @      @       @      @      @      @      @       @       @       @      (@      1@      2@      6@      1@      5@      8@      4@      8@      9@      3@      ;@      C@     �C@     �C@      J@      M@     �J@     �K@      K@     �H@     �J@     �P@      N@      O@      I@      J@     �I@     �F@      D@     �E@      E@     �B@      @@      D@      <@      H@      <@      =@      5@      8@      =@      6@      :@      9@      2@      7@      *@      1@      .@      $@      $@      @       @      @      @      @      @      @      @       @       @      @      @              �?       @      @      �?      �?              �?              �?              �?        �D��"4      9��	��`}���A"*�h

mean squared error-��<

	r-squaredRĸ>
�L
states*�L	   `���   ��@    �&A!Bl�jx7�@)�%"���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @       @      (@      @      3@     �A@      S@     �^@     �`@     �r@     P�@     ��@     d�@     @�@     �@     �@     �@     ��@     r�@     ��@     ��@     ,�@     8�@     Z�@     ��@     ��@     �@     "�@     R�@     ��@     �@     �@     ��@     f�@     ڭ@     ��@     ��@     ��@     ��@     ��@     q�@     $�@      �@     5�@     ��@     �@     }�@     5�@     �@     ��@     �@     ��@     ϶@     ��@     ��@     �@     ��@     �@     "�@     D�@     ӹ@     Ƹ@     �@     q�@     �@     �@     p�@     5�@     �@     $�@     :�@     �@     \�@     ��@     d�@     ��@     0�@     \�@     �@     �@     ��@     T�@     D�@     `�@     �@     x�@     H�@     ��@     x�@     ��@     ��@     ��@      �@     ��@     (�@     ��@     ��@     H�@     ��@     `�@     X�@     �~@     `�@     �@     ��@     Ђ@     �}@     �|@     p}@     �|@     �|@     Pz@     {@     0z@     @x@     x@     �x@     Pv@      w@     `u@     �w@      t@     @v@     �u@     `u@     r@     Ps@      s@     �u@     `s@     �s@     r@     �s@     �r@     �r@     �p@     �p@     �p@     �r@     �p@     `p@     �n@      o@     �q@     @l@     @o@     �m@      q@      k@     �j@     @k@     `k@      k@     `k@     �l@     �m@     �n@     �h@     �g@     �f@     �e@     @i@     �g@     �h@     `e@     �i@     �h@     `h@     �d@      d@     �g@     `e@      e@      d@      f@      d@      a@     �a@     �n@     @h@      b@     �`@     �`@     �b@      `@     �_@     �m@      f@     �^@     @]@      _@     @]@     �`@      _@     �`@     �Y@      _@     @`@     @Y@     �\@     �[@     �W@      a@      k@     @\@     @Z@     �]@     �U@      [@     �W@     �V@     @Y@     �Y@     �T@     �U@     �U@     �X@     �W@     �W@      U@     �U@      X@     @W@     @V@     �Q@     �R@      T@      S@      R@     @R@     @S@     @V@      U@     �Q@     �Z@     �Y@     �T@     �P@     @R@     �M@     �P@     �F@     �I@      N@      I@      N@     �K@      H@     �K@     �J@      K@     �L@      P@      G@     �I@      J@      G@      K@      C@      C@      G@     �H@     �C@      >@      C@     �B@      E@      8@      I@     �F@      7@     �A@      B@      A@      C@      >@      @@     �@@      @@     �C@      <@      ?@      ?@      @@      8@      ;@      >@      ;@      =@      >@      B@      9@      *@      <@      2@      ?@      6@      9@      0@      8@      5@      ;@     ��@     H�@      7@      9@      .@      4@      8@      6@      0@      5@      5@      6@      7@      .@      2@      >@      6@      A@      7@      <@      :@     �A@      >@      8@      <@      @@      C@      D@     �@@      H@     �@@      ?@     �C@     �D@      6@     �A@      =@      C@      C@     �C@      D@      F@     �D@      :@      H@     �A@      D@      C@      C@      E@     �D@      E@      A@      O@      I@     �H@      F@      P@     �E@      D@      I@     �E@     �A@      J@     �J@      H@     �L@      J@      J@     �P@      N@     �I@      L@     @P@     �N@      O@     �P@      L@      K@     �K@     �L@      O@      Q@     �O@      R@     @U@      T@     �R@      S@      U@     �R@      W@      R@     �R@      U@     �T@     @U@     �S@      V@     �W@     �W@     �T@      W@     �V@     �W@     �V@      V@      W@     �Z@     �]@     �\@      _@      X@     �T@      Z@     @\@      ]@      ]@     �b@      ^@     @\@      a@     @_@     @\@     �`@     �b@     @^@      b@     @_@     ``@     �b@     �b@     �e@      d@     �q@     `i@     �c@     �`@      e@     �d@     @d@      d@     @d@      e@     `c@     �d@     �e@     �b@      f@      f@      f@      e@     `j@      k@     @j@      k@     @i@     @i@     �g@     �l@     @k@     @i@     �k@     @n@     �o@     `n@     �p@     �o@     `m@     @o@     �o@     �p@     �q@      v@     �t@     �t@     `s@     0u@     �w@     �u@      s@     �v@     pu@     Pu@     �x@     �y@     @~@     �{@     v@     @x@     �~@     `@     �{@     �z@     �z@     �{@      {@     @@     (�@     Ȁ@     ��@      �@     0�@     ��@     `�@     �@     �@     ��@     ؄@     H�@      �@     p�@     ��@     @�@     �@     ��@     ��@     ��@     �@     ��@     x�@     �@     �@     t�@     D�@     ��@     D�@     ��@     ��@     "�@     D�@     ��@     X�@     ��@     `�@     ��@     ��@     ذ@     �@     �@     r�@     �@     ͸@     ��@     ��@     P�@     �@     �@     8�@     ��@     ��@     ��@     t�@     ·@     ��@     �@     ��@     ��@     /�@    ��@     V�@     �@     P�@     t�@     �@     ��@     �@     w�@     ��@     ��@     ��@     �@     L�@     ��@     �@     �@     H�@     �@     ��@     �@     �@     Ȧ@     t�@     ��@     Щ@     Ԭ@     ɰ@     Y�@     `�@     h�@     �@     ��@      �@     ��@     �@     h�@     (�@     ��@     X�@      n@     �K@      D@      =@       @        
�
predictions*�	   ��"�   �Cj@     ί@!  0!Π@)�<(�R@2�\l�9⿰1%��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G �E��a�Wܾ�iD*L�پ��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@{2�.��@!��v�@�������:�               @              �?       @       @      @      @       @      "@      @      2@      4@      <@      ;@      G@      ;@      F@      L@      M@     �Q@     �Q@     �S@      S@      V@      R@      S@     �V@      O@     �V@     �R@     �Q@     �R@     �R@      U@     �Q@     �Q@     �O@     �O@      L@      M@     �P@     �J@     �C@     �C@      =@     �C@      >@      <@      4@      3@      7@      >@      2@      ,@      0@      &@      *@      (@      $@      @      @       @      "@      @       @      @      @       @       @       @      @      @       @       @      @      �?      �?      �?              �?      �?       @       @      �?      �?              �?              �?              �?              �?              �?              �?               @      �?       @              �?              �?              �?      �?      �?               @       @      @      �?      @       @      �?      �?               @      �?      @      @      �?      @      �?      @      @      @      @      �?      @      @      $@      ,@      @      @      1@      $@      *@      $@      (@      9@      ,@      4@      2@      ;@      B@      =@      B@      D@      B@      J@      D@      G@      ?@      L@     �K@      F@      ?@      E@     �F@      >@      >@      B@      <@      =@      C@      6@      6@      ;@      9@      =@      A@      C@     �@@      ?@      7@      4@      9@      1@      &@      *@      *@       @      1@       @      &@      "@      @      @      @      @      @      @       @       @      @      @       @              �?      �?      @       @      �?      �?      �?              �?              �?        �0��23      ��	��s}���A#*�f

mean squared error9��<

	r-squared�?�>
�L
states*�L	   @K��   @3�@    �&A!^�62P�@)g�qU��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      $@      .@      9@     �D@      Q@     �X@     �f@     �x@     ��@     ��@     E�@     �@     2�@     ��@     �@     8�@     ��@     H�@     ^�@     D�@     j�@     �@     $�@     ޤ@     ��@     B�@     X�@     ܤ@     �@     ��@     B�@     0�@     ��@     �@     ��@     <�@     �@     ��@     �@     ��@     ��@     �@     H�@     ��@     <�@     1�@     ��@     ��@     �@     ն@     �@     �@     g�@     ��@     �@     ��@     ܸ@     �@     �@     ��@     ϸ@     �@     h�@     2�@     ��@     �@     �@     ��@     ��@     ��@     �@     ��@     Ҡ@     ��@     T�@     d�@     ��@      �@     ԓ@     8�@     �@     �@     Đ@     ��@     8�@     @�@     ��@     ��@     Ȇ@     ��@     �@     x�@     Ї@     0�@     �@     ��@     Ѓ@     H�@     ��@     8�@     �@     (�@     �@     ��@     �{@     0|@     0{@     p|@     �|@     0}@     �z@      z@     �x@     Py@     �w@     pw@     �w@     @v@     `u@     0v@     �v@     �s@     0u@      s@     Pt@     �s@     `s@     �r@     �r@     �r@     pq@     �p@     �q@     �p@     �p@     �p@     �p@      n@     p@     �n@     `o@      o@     �l@     `n@     �m@     �k@     `i@     �m@     �k@     @h@      i@     `k@     �g@     �i@     �i@     �i@     �m@      j@      m@     `i@     `h@     `f@     @h@     �i@     @f@     �e@      g@      e@     `c@     �e@      e@     �d@      e@     �a@     @c@     �d@     �f@      b@     `c@     �h@      j@      c@     @b@     @b@      a@      b@     �m@     @b@     @^@     �`@     �a@     �\@     @Z@     �]@     �X@     �X@     @Z@     @_@     @Z@     ``@      _@     �[@     @_@     �k@      \@     �Y@      W@     �[@     @V@     @Z@     �S@     @W@     �S@     @W@      X@     @V@     @Y@     �V@     �Q@      U@     �X@     �S@     �P@     �S@      U@     �Q@     �R@     �S@     �V@     �P@      V@     @V@     �S@      T@      N@     �[@      X@     @S@      T@      J@      M@      O@     �K@     �P@     �K@      J@      M@     �Q@     �Q@      L@     �M@      J@     �D@      J@      O@     �C@      M@      A@     �G@     @Q@      F@     �H@      F@      I@     �G@      F@     �@@     �E@     �B@      G@      C@     �C@      ?@      @@      <@     �D@      =@     �@@      :@     �@@      B@     �A@      ;@      >@      6@      =@      8@      ?@     �B@     �@@      7@      :@      7@      6@      >@      9@      "@      >@      2@      9@     ��@      �@      3@      .@      4@      3@      7@      :@      7@      ;@      :@      @@      0@      <@      ?@      8@     �@@      8@      <@      ?@      :@     �B@      ;@      1@      <@      @@      A@     �@@      =@      A@      H@      @@      @@      J@     �@@     �C@     �F@     �@@     �D@      D@     �C@      E@      B@     �D@     �F@      C@     �C@      B@     �I@     �G@      B@     �E@      H@     �D@      H@     �I@      J@     @R@     �I@      M@      G@      G@     �H@      L@      N@     �M@      M@     �P@      R@     �O@     �P@      R@     �J@     �O@     �M@     �Q@      R@      Q@      M@     @Q@     @R@     �Q@      Q@     @S@     �S@     �P@     �S@     �T@      N@     �S@     @U@     @S@      R@     �S@      Y@     @S@     �U@      Y@      Y@     �V@     �T@     �X@      Y@     �W@     �Y@     �]@     �U@      \@     �[@      a@     �\@     �]@     @\@     �_@     @[@     �[@     @`@     @]@     ``@     �]@     @^@     @`@      a@     �`@     @`@     @a@     �a@     `d@      g@      b@     �b@     �b@     @d@     `g@     @p@      e@     �b@      f@     `a@      c@     �e@     @f@      f@     `d@     �e@     �d@      h@     @g@     �g@     �e@     �e@     �h@     �h@     �i@     `i@     �i@     �k@     `m@     `i@      k@     �j@      l@      m@     �l@     �n@     �l@     pp@      p@     �n@     Pr@     �n@      q@     �t@     �u@     �t@     0s@      t@     �s@     �v@     0u@     �t@     �u@     �u@     �u@     �w@     P|@     �@     `z@     �x@     �w@     py@     �y@     @z@     p{@     `}@      }@     �~@     �@     p~@     @@     ��@     P�@     P�@      �@     ��@     H�@     @�@     �@     ��@     ��@     P�@     ��@     ��@     ��@     ��@     H�@     ��@     ��@     ,�@     ,�@     <�@     В@     �@     ��@     ��@     l�@     ��@     ��@     ��@     �@     l�@     �@     ��@     ��@     ��@     ĭ@     ��@     �@     �@     �@     �@     ��@     �@     `�@     B�@     �@     ��@     h�@     v�@     )�@     P�@     ��@     �@     ��@     E�@     ��@     ��@     j�@     2�@     �@     K�@    �o�@     �@     ��@     ��@     ��@     g�@     )�@     ��@     ��@     T�@     B�@     ʪ@     �@     V�@     &�@     ئ@     ҥ@     ��@     �@     �@     *�@     Ц@     ��@     B�@     6�@     ��@     ��@     ��@     ��@     ĳ@     ��@     ��@     X�@     ��@     H�@      �@     ��@      ~@      r@      N@      >@      C@      @        
�
predictions*�	   �gٿ    [�@     ί@!  �U��1�)����O@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��T7����5�i}1���d�r�x?�x��O�ʗ��>>�?�s��>x?�x�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�DK��@{2�.��@�������:�              �?       @       @      @      @      @      @       @      0@      9@      7@      8@     �@@      G@     �D@      G@     �H@     �D@     �P@      Q@      S@     �T@     �T@      S@      U@     �P@     �U@      R@     �Y@     �S@      U@     �Q@     @U@     �R@      R@     �O@     @Q@     �H@     �K@     �I@     �L@     �I@      K@      C@     �A@     �A@      ;@      <@      9@      5@      <@      3@      $@      .@      ,@      *@      .@      (@       @      @      @      @       @      @       @      @      @       @       @       @      @      @      @      @       @      @       @              �?      �?       @      @      �?      �?      �?              �?              �?              �?              �?      �?      �?              �?              @               @      �?              �?      �?      @              �?      @      @      @      @      @      "@      @      @      $@       @      &@      @      @      0@      .@      7@      5@      $@      (@      4@      6@      4@      ?@      ;@     �@@     �D@      ;@      B@     �A@      E@      >@      A@      C@     �E@     �G@      D@     �F@     �G@      E@      E@      @@     �@@      7@      9@      8@      ?@      >@      =@      1@      1@      5@      0@      2@      5@      *@      9@      (@      1@      &@      *@      @      @      *@      @      $@      $@      @      @      @      �?      @      @      @      @       @       @      �?       @      @       @       @      �?              �?      �?              �?              �?        �Qc�3      �3�	$��}���A$*�g

mean squared error4;�<

	r-squared@��>
�L
states*�L	   `,��   ���@    �&A!����H�@)��UкD�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      @      *@      5@     �B@     �X@     �a@     @k@      y@     ��@     <�@     7�@     �@     ԩ@     �@     ��@     ��@     ��@     �@     V�@     �@     �@      �@     ��@     "�@     ��@     �@     �@     T�@     @�@     r�@     ܩ@     4�@     ��@     A�@     Ų@     D�@     ��@     ��@     =�@     �@     P�@     ��@     n�@     ��@     ��@     ��@     c�@     ҹ@     �@     F�@     p�@     {�@      �@     6�@     ��@     7�@     �@     ��@     "�@     �@     ��@     #�@     .�@     ��@     c�@     ��@     6�@     Z�@     z�@     .�@     ��@     0�@     V�@     L�@     ��@     ��@     ,�@     ؖ@     ��@     ��@     ��@     $�@      �@     d�@     p�@     ��@     @�@     �@     �@     ��@     0�@     `�@     І@     Ȇ@     �@     ��@     ��@     8�@     0�@     ��@     `�@     ��@     h�@     �|@     ��@     ��@     �y@     {@     �z@      |@      {@     �y@     0v@     Pz@     @v@     �w@     0x@      v@     �t@     �v@     �s@     �s@     0u@     �r@     �u@     �r@      s@     �t@     �s@     pq@     �q@     �q@      q@      p@      p@     �p@     �n@     �o@     p@     �m@     @o@     �m@     @m@      l@     �m@     `m@     @m@      k@      j@     @j@     �m@     @g@     @n@     �j@     �f@     �i@     �i@     @k@      n@     @j@     �c@     �h@      g@      h@     �h@     �e@     �g@      g@     �g@     �e@     @e@     �c@     �f@     @b@     �c@      d@      `@     �b@     �`@     �a@     �d@     �_@     �k@      f@      a@     `b@      e@     �l@     �]@     �^@     `b@     �]@     @^@     �X@     @[@     �Z@     @[@     �]@     @_@      ^@     @[@      ]@     �\@      c@     �f@     �\@     @_@      V@     @T@     @X@     �W@      X@     �V@     �W@     @T@     �V@      Z@      X@     �Y@      U@     �S@      T@     @W@      S@     �P@     @X@     �P@     �S@      S@      S@     �S@      R@     @U@     �R@     �R@     @U@     @U@     �Z@     �U@      S@     �P@     @R@     �G@      P@     �P@     @U@     �P@      J@     �H@     �H@     @P@     @P@      K@      J@     �E@      K@     �H@      G@      I@      H@     �K@      F@      E@      G@      L@     �K@      H@      D@     �E@     �D@     �E@      D@      C@     �C@      E@      D@      A@      >@      C@      @@      ?@      <@      =@     �D@      C@      ;@     �A@      =@      <@      :@      ;@      8@      ?@      3@      3@      >@      =@      9@      9@      8@     �@     �@      9@      ?@      >@      4@      ;@      ;@      3@      1@      0@     �@@      :@      >@      8@      A@      ?@      :@      =@      =@      ;@     �@@     �A@      B@      9@      G@      B@     �E@      A@      A@      F@     �B@     �I@     �B@     �C@      =@     �D@     �H@     �@@      G@      <@      A@      C@      F@     �F@     �G@      E@     �E@     �H@      F@     �L@     �E@      E@     �G@      F@      M@     �K@     �L@     �J@     �L@     �K@     �L@     �I@      M@      L@     �L@     �P@      I@      L@      T@      S@      P@     �Q@     @Q@      V@     �L@     �Q@     �O@      T@     �R@      S@     @P@     �T@     @R@     @V@      Z@     @X@     �U@     �X@     �R@     �R@     �V@     �T@     @V@     @X@     �S@      W@     �S@     �T@     @R@     �V@     �X@     �U@     �[@     �Y@     �X@      [@     �Z@     �\@     �]@      ]@     ``@      `@      _@     @_@      _@     �`@     �`@      ]@      a@     �_@      _@     �_@     `b@     �a@     `a@     @g@     �d@     �b@      d@     �`@     �a@     �d@     �j@     0q@     �b@     `a@     �a@     �c@     �c@     �f@     �c@     �f@     �a@      e@     �e@      i@     �f@     �k@     `f@     �i@     �h@      h@     �f@     �h@     �m@     `i@     �i@     �j@      k@      n@     �j@      l@     `l@     �l@      m@      p@     �p@      l@     p@     �q@     s@      v@      t@     �r@     @q@     `u@     Ps@      t@     pv@     `v@     �{@     Py@      w@     w@     @v@     8�@     `z@     0y@     �z@      y@     �z@      |@      y@      {@     ��@     0@     ~@     �@     p@     @@     �@     Ȁ@     p�@     �@     �@     �@     ��@     ��@     H�@     І@     P�@     ؈@     ��@     Ќ@     X�@     0�@     �@     $�@     �@     T�@     @�@     h�@     P�@     P�@     ܚ@     $�@     Ȟ@     ��@     H�@     ڣ@     ؤ@     �@     �@     �@     z�@     ��@     ��@     #�@     r�@     )�@     \�@     �@     ��@     E�@     y�@     ^�@     ;�@     ��@     }�@     ^�@     �@     ��@     �@     o�@     �@     1�@     ǻ@      �@     ޾@     "�@     %�@     ��@     ��@     ��@     ��@     T�@     ��@     	�@     ��@     ��@     Z�@     z�@     b�@     ��@     ��@     ��@     Ʀ@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     �@     ��@     u�@     g�@     W�@     ��@     h�@     �@     ��@     �@     x�@     Ѓ@     �n@      J@      B@      =@      "@        
�
predictions*�	   ���տ   ���@     ί@!  �M�_H@)�3�QS@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
��ѩ�-߾E��a�Wܾ��(���>a�Ϭ(�>�ߊ4F��>})�l a�>��[�?1��a˲?>h�'�?x?�x�?��d�r?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�               @       @      @      @      @      $@      (@      ,@      1@      3@      =@      <@      A@     �F@      G@     �J@     �M@     �M@     �K@     �M@      Q@      S@     @U@      M@      N@     �P@     �M@     �O@     �P@      R@      H@     �P@     �I@     �G@      H@      F@      H@     �J@     �A@     �A@      =@     �A@      4@      >@      8@      :@      1@      .@      2@      .@      (@      0@      0@      *@      .@      @      $@      @      @      "@      "@      "@      $@      @      @               @      @              @       @       @       @      �?      �?      �?       @              �?      �?       @       @       @              @               @              �?              �?              �?              �?              �?              �?              �?       @              �?       @              �?      @      �?              @              �?      @      @      @      @      @               @      @       @      @      @      @      @      (@       @       @      &@      (@      8@      5@      ,@      ,@      (@      3@      1@      4@      <@      ,@      8@      >@     �@@     �B@      C@      I@     �I@      K@      M@     �J@      G@     �R@     �L@      P@     �M@     �I@     �N@     �Q@      F@     �J@      D@     �A@     �D@      ;@      E@      <@      <@      2@      C@      :@      5@      6@      5@      ;@      1@      4@      4@      1@      0@      2@      &@      .@      4@      3@      @      @      @      @      @      @               @              @      �?      @      @      �?      @      �?      �?      �?              �?      �?              �?        ��%��3      �3�	�8�}���A%*�g

mean squared errorT%�<

	r-squared���>
�L
states*�L	   `���   @*�@    �&A!@<��a�@).��#3 �@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      $@       @      >@      O@     �R@     �`@     �g@     �v@     ��@     �@     �@     d�@     ʫ@     !�@     z�@     ��@     ��@     x�@     V�@     n�@     X�@     С@     F�@     �@     ��@     b�@     �@     �@     ��@     Z�@      �@     ��@     L�@     ʰ@     x�@     S�@     V�@     ظ@     ��@     y�@    �?�@    �_�@    �O�@     �@     ̽@     ��@     ��@     ��@     N�@     ��@     ��@     i�@     ��@     ϵ@     �@     /�@     ?�@     ��@     Z�@     ��@     	�@     %�@     ��@     B�@     ��@     _�@     Ԭ@     �@     ��@     ��@     H�@     v�@     �@     �@     �@     d�@     �@     �@     ��@     �@     �@      �@     �@     ��@      �@     ��@     ��@     ��@     ��@     ��@     p�@     ��@      �@      �@     ��@     �@     ؂@     ��@     X�@     �@     �@     �@     �~@     �~@     p{@     �{@     �}@     {@     �x@     �{@     �z@     �x@     @w@     `x@     �x@     �v@     `x@     pu@     pv@     �v@     �s@     `u@     �t@      s@     s@     0u@     0s@     �r@     �r@     �q@     0r@      p@     @p@     �o@     0q@     `o@     0q@     p@     �n@     @p@      n@     @m@     �n@     �l@     @l@      k@     `m@     �l@     �l@     �j@     `k@     @k@      g@     �h@     �i@      i@     �l@     @f@     �g@     `h@     @m@      k@     `f@     �b@     �g@     �c@     `e@     �f@     `d@     �i@     p@     @j@      f@     �d@     �d@     `a@     �a@      b@     `d@     �c@     �a@      a@      `@      [@     @^@     `a@      ]@     @a@     @o@     @a@     @_@     @_@      a@     @\@     �`@     �]@     @_@     �\@     �\@     �`@     �W@     �W@     �[@      \@     `j@     `b@     �^@     �W@     �X@     �V@     �R@     �T@     �T@     �V@      T@     �R@      U@     @Y@      R@     @X@     �U@      S@      U@     �Q@     �Q@     �S@      T@     �S@     �O@     @W@     �R@      P@      Q@     �T@      T@      S@     @R@      T@     �U@     @W@     �S@     �R@     �S@     �S@      J@     �I@      R@     �I@     �M@     �J@      L@     �I@      K@     �F@      I@     �G@      K@      J@     �F@      N@      G@      I@      F@      I@     �J@     �B@      I@      F@      A@     �B@     �C@     �B@     �E@     �C@      J@      ?@     �B@      <@     �D@      B@     �@@     �F@      ?@     �@@     �D@     �D@      5@      >@      8@      >@      A@      =@      :@      2@      C@      8@      @@      9@      .@      =@     ��@     �@      0@      <@      7@      6@      A@      =@      @@      ?@      ;@      9@      >@      @@     �A@      >@     �B@      8@      >@      @@      2@      :@      ;@     �E@      B@      @@     �@@     �C@      D@      9@      D@     �D@     �E@      M@      O@      G@      @@      K@      F@     �F@     �I@      K@      @@     �K@      G@     �J@      G@      F@     �C@      F@      J@      L@     �K@     @P@      I@      J@      S@     �H@      O@      I@      N@     �K@     �P@      O@     �L@      O@      I@     @Q@     �T@      V@     @Q@      R@     @P@     �S@     �P@     �S@     �P@     @R@     �R@     �R@     @S@     �V@      P@      N@     @P@     �W@      P@     �Y@      X@     �W@      V@      Z@     �U@     �W@      Y@     �V@     @V@     �[@     @W@      W@     �]@     �Z@      `@      Z@     �U@     @Z@     �Z@     @[@      _@     �`@     �X@     �`@     @]@     �_@     @[@     �\@     @]@     @_@      b@     �c@     @_@     �a@     �_@     �`@     @c@     �h@     `b@     @b@     �c@      a@     @b@     �b@     �a@     �f@     Pp@     `d@     @e@     �d@     �d@     @d@     �g@     �f@     �f@     �h@      d@     �f@     `i@     �d@     �g@      k@     @j@     @i@      k@     @l@     �j@     �k@     �l@      n@      j@      m@     �m@      p@     �n@     �p@     `o@     �n@     �l@     @p@     Pr@     �u@     0u@     �r@     �v@     w@     Pu@      t@     0t@     �t@     �s@     �u@     `u@     pv@     �t@     Pw@     �w@     �x@     ��@     @w@     �x@     �x@     �z@      |@     �{@     0|@     0{@     �y@     p}@     ��@     �}@     �@     �~@     P�@     x�@     �@     ��@     H�@     ��@     ��@     ��@     ��@     ��@     �@     ȋ@     �@     ؊@     �@     ��@     ؎@     ��@     8�@     ��@     ܒ@     ؔ@     ��@     L�@     L�@     Й@     �@     �@     ��@     �@     ��@     �@     �@     ��@     4�@     �@     ��@     ܲ@     �@     
�@     ^�@      �@     C�@     ں@     m�@     ��@     �@     ҷ@     ��@     p�@     t�@     ��@     ��@     ж@     �@     ׺@     j�@     �@     ��@    �+�@     r�@     ��@     \�@     }�@     ,�@     7�@     (�@     Y�@     �@     ��@     �@     ��@     ��@     ��@      �@     ��@     Ħ@     n�@     8�@     ��@     ƥ@     �@     R�@     ��@     ��@     \�@     ��@     ��@     T�@     	�@    �2�@     �@     ��@     �@     ��@     X�@     ��@     (�@     Pt@      L@      H@     �A@      (@        
�
predictions*�	   ��Y�    �I@     ί@!  �XG�/�)��	L��R@2�+Se*8�\l�9⿰1%���Z%�޿��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'���ߊ4F��h���`�K���7��[#=�؏���XQ��>�����>a�Ϭ(�>8K�ߝ�>f�ʜ�7
?>h�'�?�vV�R9?��ڋ?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�6v��@h�5�@!��v�@زv�5f@�������:�              �?              �?              �?              @      @       @      "@      @      *@      7@      =@     �D@      G@      I@     �D@     �K@      L@     �R@     �P@      S@     @U@     �R@     �P@      V@     �W@      Z@      X@     �T@     �Y@     @V@     @T@      U@     �U@      V@     @U@     �X@      M@      K@      O@     �K@      G@      A@     �G@     �B@      <@      :@      8@      4@      3@      6@      1@      1@      5@      &@      .@      $@      "@       @      ,@      @      &@      &@      @      @      �?      @       @      �?      @      @       @      @              �?      @       @      @       @              �?      @      �?              �?       @      @       @              �?      �?              �?              �?              �?              �?              �?               @              �?              @       @      �?       @              �?      @              �?      �?               @      @      @       @      @       @      @       @      @       @      @      &@      &@      (@       @      1@      0@      (@       @      *@      *@      7@      8@      :@      8@      7@      ?@      8@      A@     �C@      @@     �A@     �@@      @@     �C@      @@      D@     �B@      @@      ?@      8@      >@      =@      <@      7@      @@      9@      ,@      =@      2@      .@      5@      <@      :@      *@      1@      0@      ,@      *@      "@      (@      0@      0@      "@      $@      $@       @       @      @      @      @       @      @      @      �?      @       @              @      @      �?               @              �?              �?              �?        I�ݒ3      Y2Y	��}���A&*�g

mean squared errorJ,�<

	r-squaredV��>
�L
states*�L	   �m��   `��@    �&A!f�Z|��@)����(�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      @      &@       @      8@      R@     ``@     @a@      e@     �r@     x�@     <�@     �@     ��@     �@     ϱ@     j�@     ��@     ��@     ^�@     �@     ��@     n�@     b�@     8�@     �@     ~�@     �@     �@     ��@     z�@     >�@     ��@     ��@     ��@     )�@     �@     �@     �@     Y�@     �@     $�@     ¿@     C�@     -�@     ��@     ׼@     ڻ@     s�@     ��@     8�@     (�@     ��@     _�@     ��@     A�@     ߵ@     1�@     ظ@     ��@     L�@     �@     ��@     ��@     {�@     ��@     ,�@     �@     ��@     �@     &�@     ��@     X�@     ��@     �@     ��@     ��@     <�@     ,�@     ��@     ��@      �@     ̒@     �@     h�@     p�@     p�@     ��@     8�@     ��@     ��@     X�@     ��@     ��@     P�@      �@     ��@     ��@     ��@     ��@     p�@     @�@     @�@     ��@     �|@     �}@     `}@     �|@     p}@      }@     �|@     �y@     p{@     Py@     �x@      x@     �x@     x@     `w@     `w@     `v@     0u@     �w@     Pt@     �t@     s@     pt@     �t@     �u@     Pq@     q@     @q@     pr@     �r@     pq@     r@     Pp@     �p@     �o@     q@     pp@      o@     �o@     `l@     �p@     @l@     �k@     �l@     @m@     �i@      i@     �l@     �n@      h@     �i@     �i@     �j@      k@     �g@     @h@     �p@     q@     �j@     @f@     �n@     @j@      d@     �f@      e@     `e@     �e@     �e@      d@     �b@      d@      d@     �`@     �e@     �c@     @^@     �b@      b@     �b@     ``@     �b@     �`@     �`@     �a@     `a@     �]@     �g@     �f@     �^@     �`@     �]@     �\@      ]@      ]@     @W@      Y@      ^@     @Z@      Y@      \@     �[@     �Z@     �^@     �m@      Z@     �Y@     �Z@      [@     �W@      [@     �W@      U@     �Y@     @X@     �R@     �U@     �T@      U@     �U@      V@     �Q@     �R@     @Z@     @T@     @W@      T@     @T@     @T@     �Q@     �Q@     �U@      U@      Q@     @P@      Q@     @Q@     �S@     �T@      M@     @]@     �S@      I@      F@      R@     �L@     �G@     �L@     �G@      I@      O@      J@      F@      G@      I@      H@      O@      G@      A@     �K@      E@      D@     �F@      B@     �D@     �D@      G@      D@     �@@      E@     �C@     �G@      >@     �A@      ;@     �E@      C@      =@      9@      B@      G@      >@      H@      9@     �D@      A@      4@      >@      <@      B@      =@      @@      7@      7@      <@      =@      :@      7@      .@     ȗ@     ��@      9@      2@      9@      >@      6@      9@      6@      <@      8@      >@      ;@      A@      C@      ?@      =@     �E@      5@     �A@      @@      6@     �@@      =@      7@     �C@      A@      9@      @@      @@      G@     �D@      J@      G@      ?@     �F@      D@     �D@     �A@     �H@     �D@      D@      C@      F@     �L@     �J@      D@      L@      I@     �J@     �J@      D@      P@      H@      J@     �D@     �I@     �J@     �D@     �N@     @Q@     �P@      K@     �N@     �O@     �P@     �P@     @P@      O@     @R@      K@     �K@     �L@     @P@     �P@     �M@     @R@      S@     �S@     �T@      S@      Q@     �T@     �S@      Q@     �P@     �Q@     �U@     �U@     �Q@     �W@     @T@      V@     @V@     @T@     �V@      Y@     @W@      Y@     �Y@     @W@     @X@     @X@     �X@     @Y@      Z@     �]@     �\@     @\@     �]@     �X@      ]@      [@     �a@     �]@     �^@     @\@      [@      `@     �`@     �`@     �a@     �i@     �a@     @`@      a@     �a@     @d@     @a@     �c@     �c@     �e@     `b@     �e@      r@     @b@      b@     �b@     �e@     �b@      c@     �d@     �e@      j@     �i@     @g@     `f@      j@     `f@      h@      h@     �g@      i@     �f@     `k@      j@     �j@     �i@      k@     `j@      k@      m@     �m@     p@     pp@     �o@     �q@     pp@     �u@     �z@     @x@     �r@     r@     pr@     �q@     0r@     v@     �t@     Pv@      t@     `s@     �x@     x@     �t@     �w@     0x@     ��@     �y@     �w@     Px@     Py@     �y@     �x@     �z@     }@     �{@     8�@     }@     �~@     �~@     0�@     ��@     ؀@     ؁@     0�@     @�@     h�@      �@     P�@     ��@     H�@     ��@     ؈@     @�@     ��@     @�@     �@     0�@     ��@     \�@     ��@     ��@     ��@     ̖@     ��@     �@      �@     �@     (�@     Р@     �@     2�@     :�@     ¨@      �@     �@     ܰ@     ��@     ��@     �@     ��@     ]�@     n�@     `�@     �@     ͹@     1�@     ۸@     &�@     /�@     �@     W�@     ��@     �@     :�@     ��@     ͹@     ��@     H�@     '�@    �A�@     ȿ@     ̽@     ��@     ӻ@     ��@     u�@     ʲ@     �@     J�@     D�@     ,�@     h�@     �@     (�@     >�@     ��@     ܥ@     ¦@     v�@     6�@     ��@     ¦@     ��@     ��@     ��@     ��@     ��@     +�@     ��@     h�@     6�@     �@     x�@     x�@     �@     ��@     �@     �}@     �t@     �^@     �@@      C@      @        
�
predictions*�	    �տ   @�@     ί@!  0+w�$@)+�<e_�O@2���7�ֿ�Ca�G�Կ_&A�o�ҿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.��>h�'��f�ʜ�7
��h���`�8K�ߝ�����>豪}0ڰ>�h���`�>�ߊ4F��>6�]��?����?>h�'�?x?�x�?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@{2�.��@!��v�@�������:�              �?      �?              @      �?      "@      (@      (@      6@      ;@      <@     �A@     �E@      I@      L@      H@      P@      Q@     @R@     �S@     @U@      X@      W@      Z@     @Z@     @X@     �U@     �Q@      V@     �V@     @R@     �P@      U@     �Q@     �S@     �I@      G@      G@      H@     �E@      C@      >@      B@      =@      9@      4@      =@      (@      0@      $@      4@      $@      ,@      ,@      *@      &@      $@      "@       @      @      @       @      @      @      @      @       @      @      �?       @      @      �?       @      �?              �?      @      �?       @       @              �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?       @      �?      �?               @      �?               @       @      �?      �?      @      @      �?      @      @      @      @      @      @      �?      @      @      "@      $@      *@      $@      $@      0@      .@      .@      0@      (@      6@      .@      8@      6@      A@      @@      C@      =@      ?@      7@      @@     �A@      D@     �D@      I@     �F@     �@@     �C@      C@      B@      :@      <@      >@     �B@      :@      ;@     �C@     �A@      A@      :@      <@      =@      7@      5@      0@      <@      ,@      7@      6@      0@      *@      .@      @      @      @      &@       @      @      "@       @      @       @      @      @      �?      @      @       @              �?              �?      @              �?              �?        �=S�3      �3�	�c�}���A'*�g

mean squared error
�<

	r-squared�l�>
�L
states*�L	   ����    ��@    �&A!��#}!U�@)����8n�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      $@       @      .@      :@     @T@     �]@     ``@     �d@      t@     x�@     ��@     g�@     ��@     ث@     +�@     ��@     ��@     `�@     ��@     �@     B�@     ��@     ��@     �@     �@     .�@     ��@     �@     �@     �@     >�@     R�@     h�@     ��@     Ʈ@     ߱@     �@     h�@     ^�@     ��@     5�@    ��@     Կ@     �@     ��@     �@     ��@     �@     ��@     �@     ��@     9�@     �@     �@     |�@     ö@     ��@     ��@     ��@     ��@     �@     F�@     շ@     9�@     ��@     ��@     K�@     ګ@     j�@     N�@     `�@     J�@     ��@     ��@     t�@     ��@     <�@     �@     <�@     `�@     ��@     L�@     ��@     8�@     ��@     8�@      �@     ��@     (�@     P�@      �@     �@     �@     ��@     �@     ��@     0�@     �@     ��@     H�@     Ё@     ��@     ��@     �}@     �}@     p~@     �}@     P}@     p{@     `x@     @}@     �x@      {@     �z@     �x@     �w@     �x@     @w@     `w@     Px@     �u@     `v@     �u@     �t@     �u@      v@     �t@     �r@      t@     �s@     �q@     �r@      r@     @p@     0q@     �r@     �q@      q@     �n@     @p@     `m@     �p@     �o@     �l@     �j@      n@     `k@     @n@     �h@      m@     �n@     `k@     �j@     `i@     �l@     `s@     0r@     @j@     �h@     �j@     �f@     �j@      f@     `d@     �e@     �m@     �h@     �d@     �d@      f@     �f@     �d@     �d@     @c@      b@     �e@     �b@     `c@     �f@      b@     @b@      d@      c@     �`@     �a@     �`@     @b@     �a@     @^@     @d@     @k@     @_@     @_@     �_@     @]@     �`@      a@     @\@     �Z@     �^@     @\@      ^@      a@     �^@     �Z@     �]@     @[@      j@     @X@     �V@     �_@     �]@     �]@     �W@     @[@     @Z@     @Y@      V@     @V@     �W@     �Z@     @X@     �T@      U@     �S@     �T@     �T@     @T@     @W@      R@      R@      S@     @T@     @Q@     �Q@     �P@      Q@     �Q@     �Q@     �N@      O@     @Q@     �S@     �U@     @Z@     @Q@     �Q@      J@     �J@      L@     �I@      K@      L@      L@     �Q@      I@      H@     �E@     �K@      K@      J@      <@      L@      L@     �K@      ?@      D@      B@     �C@      >@     �B@     �F@      I@     �G@      H@      G@     �B@      G@      A@     �B@      C@      <@      <@     �A@      ?@      ?@      A@      >@      8@      8@      A@      :@      1@      :@      <@      6@      >@      6@      ?@      3@      =@     �@     �@      ,@      8@      >@      4@      7@      >@      6@      ;@      ?@     �A@      ?@      7@     �B@     �@@      ?@      ;@      G@      <@      8@      ;@      6@     �D@      ?@     �B@      3@     �D@      A@      >@      F@      ;@      ?@      C@      F@     �A@      D@      A@      F@      H@      @@      G@      E@      G@     �A@      F@     �F@     �F@      G@     �N@     �D@      H@     �K@     �I@     �H@     �G@      K@      K@     �Q@     �P@      K@      M@      F@     �S@     �N@     �O@      P@     �N@      M@     @P@     @R@     @S@      P@     @P@      W@     @R@      N@     �R@      T@     �P@     @Q@     @P@      P@     �Q@     �P@     @S@     �O@     �T@     �N@     @T@     @W@     �S@     @T@     @W@     �V@     �T@     @V@      \@     @Y@     �U@     �W@     @X@      [@     @Z@      \@      ]@     �]@     �]@     �Z@     �[@     �Z@     �]@     �^@     ``@     �]@     �]@      [@     ``@     �\@     @^@     �Z@      a@     �]@     @^@     @a@     `h@     `h@     @d@     �`@      c@     �b@      c@      c@     `i@     @p@      b@     �a@     @d@     �f@     �b@     �e@     @e@      e@     @e@      h@      d@      h@      g@     �c@     `c@     `g@     �i@     �k@      i@     �i@      h@     �j@     �k@      l@     �m@     �l@      l@     `k@     Pp@     @p@     q@      u@     �t@     �p@     `p@     �t@     `w@     �q@     @t@     s@     Ps@     ps@     �w@     0t@     Pu@     �u@      t@     `v@     �{@     �u@      ~@     �x@     �y@     �v@     @y@     �x@     0z@     �x@     �y@     �z@     �}@     �|@     �}@     x�@     @@     p~@     ȁ@     Ȁ@     ��@     �@     �@     ��@     `�@      �@     ��@     h�@     �@     ��@      �@     ؉@     @�@     p�@     �@     h�@     ��@     0�@     ̒@     ؓ@     �@     `�@     X�@     Ԛ@     ��@     t�@     ��@     �@     �@      �@     j�@     ª@     ~�@     �@     ��@     R�@     `�@     ط@     s�@     �@     �@     ��@     ź@     ѹ@     ��@     ʷ@     Ѷ@     ʵ@     �@     ��@     �@     F�@     7�@     ��@     ��@     ѽ@     ο@    �k�@     �@     �@     ׼@     ��@     X�@     Y�@     �@     Ӱ@     ��@     �@     ��@      �@     
�@     ��@     �@     Ʀ@     ��@     ��@     $�@     ��@     ��@     Ч@     ��@     6�@     V�@     ��@     �@     ��@     Ѵ@     i�@     ~�@     О@     ؋@     (�@     �@     (�@     h�@     �{@     �r@     0q@     �E@      ?@      *@        
�
predictions*�	   �n8�   ��'@     ί@!  h��&�)���F�1P@2�\l�9⿰1%���Z%�޿W�i�bۿ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
��������[���FF�G �O�ʗ�����Zr[v��8K�ߝ�>�h���`�>��Zr[v�>O�ʗ��>�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@�DK��@{2�.��@�������:�              �?      �?      �?               @      @      @      @      @      .@      *@      >@      A@      <@     �F@      G@     �J@     �N@     �P@      P@     @R@     �U@     @R@     @W@      W@     �U@      Z@     �Y@      [@     �X@      Z@     @V@     �X@     �Y@      J@     �Q@     �P@     �O@     �M@     �L@      J@      O@      ?@      B@     �D@      >@      <@      7@      9@      4@      0@      .@      ,@      "@      1@      $@      @      (@      3@      &@      @      &@      @      @      @      @      @      @      @       @       @              @      @      �?       @       @      @      �?       @              �?               @              �?      �?       @              �?      �?               @              �?              �?              �?              �?              �?               @      �?              �?              �?      �?      �?              �?      @       @      @      �?              @      @      �?      �?      "@      @      �?      @      @      @      @      @       @      1@      *@      (@      *@       @      ,@      8@      :@      3@      4@      :@      8@      9@      @@     �@@      ;@      C@     �A@     �@@      @@      7@      9@      6@      D@      ?@      =@      @@      B@      :@      :@     �C@      <@      1@      0@      ;@     �@@      <@      7@      7@      9@      .@      2@      1@      2@      4@      0@      *@       @      &@      &@      &@      $@      @      $@      @      (@       @       @      @      @      @      @      @       @      �?      �?      �?               @       @              �?              �?              �?        yvx��3      EN�	;�}���A(*�g

mean squared error�q�<

	r-squared�^�>
�L
states*�L	   @���    ��@    �&A!L��P��@)	IB���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @      *@      @      C@      H@     �N@      a@     `c@     @p@     0x@     h�@     ,�@     ʿ@     b�@     b�@     ��@     >�@     :�@     ��@     ��@      �@     2�@     ��@     ��@     f�@     z�@     N�@     �@     �@     4�@     B�@     ��@     ��@     ��@     T�@     ��@     ��@     ��@     �@     (�@     ��@     ��@     e�@     ��@     ��@     ��@     м@     ػ@     a�@     ��@     U�@     �@     Ƕ@     �@     8�@     �@     ��@     �@     <�@     �@     [�@     ɸ@     ��@     o�@     +�@     ��@     �@     ��@     ��@     ��@     F�@     :�@     h�@     �@     .�@     r�@     ��@     �@     ��@     ��@     ��@     4�@     �@     ܑ@     ȏ@      �@     X�@     ��@     ��@     H�@     X�@     8�@     x�@     ��@     ��@     ��@     ��@     ��@     0�@     ��@     ��@     ��@     h�@     0�@     ؁@      �@     �~@     P|@     `{@     0}@     0}@      y@     `y@     pz@     �y@     �w@     �v@     �v@     Pv@     �u@      x@     v@      w@     �s@     �t@     r@     @u@     �s@     �q@     �s@     pq@     @t@     �r@     �r@     �o@     �p@      r@     �q@     @n@     @q@     pp@     0p@     p@      o@     �n@     �l@     @p@     �l@     �m@     @n@     �s@     pq@     �n@     �m@     @i@     @g@     �j@     �j@     `g@      e@     @j@     �g@     �h@      i@      e@     @e@     �e@      f@     �f@      h@     `d@      e@     �g@      g@      i@     �a@     `d@     �b@      d@     @c@      d@     @c@     �a@      ]@     �f@     �_@      `@     �_@     �`@      `@      ]@      k@     �c@      b@     �`@     �^@     �\@     �[@      ^@     �`@     �^@     �Z@      ^@     @_@      _@      `@      ]@     �\@     �^@      \@     �^@      f@     �a@     @`@     �X@     �W@      W@     �Z@     �V@     @U@     @W@     �W@      T@      X@      X@     �T@     �R@      S@     �T@     @Q@      W@     @S@     �Q@     @S@      R@      T@      R@     �Q@      S@     �J@     �M@      I@      Q@      L@      Q@     @R@     �I@     �R@     �O@     �L@      M@      T@     @S@      I@     @P@      D@      C@      G@     �E@     �G@     �D@     �P@      G@     �G@      E@     �B@     �B@     �A@     �F@     �E@     �D@     �E@     �K@     �@@      F@     �A@      C@      E@      G@      A@      F@      ;@      E@     �@@     �B@     �B@      >@      A@      D@      C@     �@@      =@      @@      A@      7@     �A@      =@      6@      4@      :@      <@      =@     �@     H�@      3@      =@      8@      0@      2@      8@      B@      4@      9@      4@      @@     �@@      :@      ?@      9@      5@      7@      9@      .@      ;@     �@@      5@     �A@      <@      C@      7@     �A@      K@      @@     �C@      C@     �@@     �F@     �I@      D@      D@      A@      D@     �A@      C@      ?@     �A@      M@     �@@     �H@     �D@     �C@     �L@     �L@     �Q@      M@      P@      I@     �D@     �G@     �Q@      K@      I@      K@      G@      O@     �P@     �J@     �J@      G@     �N@      M@      N@     @Q@      P@      O@     �O@     �Q@      O@     �O@     �S@      S@     @R@     �R@     �U@     �R@     @P@      T@     �T@      W@     �T@     �M@     �X@     @T@     �T@      X@     �U@      T@     @U@     �O@      X@     �W@     @Z@      Z@      ]@     @Z@      Z@     �\@     �T@     �`@     �]@      \@     @\@     @^@     �\@     �[@     �[@     �[@     �X@     @Z@     �[@      `@     `b@     �^@      `@     `a@     �^@     �`@     @]@     @g@     `i@     �c@     `a@     @f@      b@     �a@      g@     �p@     @f@     �d@     @b@     @e@     @f@      f@     `f@     @f@      f@     �j@     `e@     �d@     @g@     �g@      h@     �i@     �g@      g@     `g@     �j@     �g@     �h@     �h@     �j@      l@     �l@     @p@     �n@      l@     �p@     Pq@     �u@     t@     �r@     �s@     �t@     �r@     `r@     r@     @s@      s@     @s@      u@     u@     `u@     s@     �t@     �{@     Px@     Pu@     �w@     @@     @w@     @w@     �y@     `w@     `y@     P|@      x@     P|@     0|@     �{@     �~@     �~@      @      �@     x�@     І@     ��@     ��@     0�@     �@     �@     ��@     @�@     8�@     X�@     8�@     8�@     ��@     8�@     H�@     �@     8�@     ��@     Ȑ@     ��@     l�@     �@     �@     ��@     4�@     Ț@     �@     ��@      �@     :�@     ��@     ֦@     0�@     ��@     ��@     :�@     Z�@     ݴ@     ε@     %�@     0�@     �@     ��@     ܹ@     ߸@     <�@     ��@     <�@     ��@     b�@     �@     l�@     ��@     6�@     �@     _�@     G�@     ��@     -�@     �@     Ѿ@     e�@     ��@     ��@     ��@     3�@     #�@     d�@     �@     ��@     N�@     �@     J�@     d�@     ��@     ��@     "�@     ��@     ��@      �@     "�@     >�@     ک@     N�@     Z�@     ߱@     ��@     A�@     z�@     ��@     ��@     �@     p�@     Ѕ@     ��@     P}@     w@     �p@     pr@     �K@      @@      &@        
�
predictions*�	   �`�    ֒@     ί@!  �Nq�1�)����3}Q@2��1%���Z%�޿�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9��T7���f�ʜ�7
������pz�w�7��})�l a���Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?����?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@{2�.��@!��v�@�������:�              �?              �?       @      @       @       @      ,@      ,@      (@      7@      5@      C@      @@     �I@     �I@      N@     �O@      U@     �T@     �U@     �S@     @Q@      Q@      R@     �T@     �R@     �T@     @U@     @W@     @T@     @T@      U@      V@     �S@     @R@     @U@      L@      G@     �I@      F@      D@     �D@      7@     �A@      >@      :@      6@      4@      >@      >@      ,@      7@      ,@      "@       @      *@      $@      "@      "@      &@      $@      @      @      @      $@       @      @      @       @      @      @      �?      @       @       @              @      �?       @       @              �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              @              �?              �?              �?      �?      @      �?      �?      @       @      @      �?      �?      �?      @      �?      @      �?       @      @       @      @      @      @      �?      @      @      $@      *@      &@      *@      2@      1@      3@      &@      0@      5@      8@      ;@      8@      9@      =@      ?@      F@     �E@      C@      B@     �J@      A@     �C@      D@      =@      A@      >@      B@      C@      A@      9@      >@      8@      :@      A@      5@      9@      9@      7@      6@      5@      8@      @@      ,@      1@      1@      (@      ,@      $@      (@      .@      0@      @      @      @      @      @      (@      @      @      �?       @      @      �?      @      @      �?      �?              �?              �?              �?              �?        ���R4      d��	���}���A)*�h

mean squared error F�<

	r-squared.��>
�L
states*�L	   `���    �@    �&A!�Ǵ4u�@)�n@��X�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@      .@      &@      1@      >@      R@     �\@     �f@     �k@     �v@     ��@     ��@      �@     ��@     f�@     A�@     >�@     Z�@     (�@     N�@     Ȧ@     �@     f�@     ��@     J�@     ��@     P�@     p�@     2�@     L�@     D�@     ��@     ĩ@     �@     L�@     6�@     d�@     ��@     ��@     ݶ@     ,�@     ĺ@     �@     ��@     ��@     ^�@     ��@     ��@     º@     չ@     ��@     ��@     ��@     ط@     !�@     Ķ@     3�@     ��@     ѷ@     �@     ��@     Ÿ@     ҷ@     ��@     >�@     �@     ^�@     ��@     ��@     ��@     ��@     ��@     V�@     ��@     D�@     2�@     l�@     x�@      �@     ��@     ��@     l�@     ��@     T�@     P�@     ��@     �@     x�@     �@     0�@     8�@     ��@     p�@     �@     @�@     (�@     0�@     ��@     ��@     (�@     ��@     ��@     ��@     p�@     8�@     p~@     P|@     p~@     pz@     �}@     |@     �}@      {@      y@     �y@     @y@     �x@     @z@      y@     �w@      x@     �w@     0u@     �r@     �t@     pv@     �u@     �u@     �s@     �q@     �s@     �r@     `t@      q@     �q@     �s@     �p@     `q@     �m@     s@     �o@     �q@     �n@     p@      p@     �n@     @n@     0s@     Pq@     �o@     �j@     �n@     �n@     �i@     �j@     `k@     �g@      h@      h@      i@     �e@     �i@     �g@     @h@      i@      f@     @f@     �d@     @e@     @f@     �c@      c@     �c@     �e@     �c@     �a@     �d@     `c@     �b@     �c@     `e@     @d@     �j@      b@      a@     �c@     �b@     �_@     �d@     @a@     �^@     �n@     @c@     �`@      a@     ``@      a@     ``@     @_@      \@      ]@     �\@     �a@      `@     �a@     `a@     @`@     �Z@     �\@     �\@      ^@     �Y@     �[@     �W@     @Z@     �W@      X@     @Y@     @T@      e@     @b@     @S@     �X@     @T@      U@     �R@     �R@     �V@      V@     �W@     �X@      R@      T@     @Q@      P@      P@     �Q@      Q@     @R@      Q@     �L@      N@      N@     �Q@      I@     �J@      L@     �L@     �Q@     �O@      K@      K@     �N@     �L@      P@      M@     �N@     �O@      F@     �Y@     �T@     �O@      N@     �A@     �G@     �G@      I@     �E@      K@      I@     �J@     �C@      ?@     �F@      G@     �K@     �C@      D@      :@     �@@     �B@      9@      B@      ;@      @@      E@     �B@     �C@      A@      A@      >@      D@      :@      6@      9@      A@      @@      8@      @@      ;@      ;@      >@     P�@     ��@      7@      6@      0@      3@      9@      7@      >@      7@      8@      4@      2@      *@     �B@      .@      4@      6@      ?@     �@@     �A@      >@      =@      <@     �@@      A@      ?@      B@      9@      6@      B@     �D@      B@     �A@     �D@     �E@     �B@     �C@     �E@     �C@      I@     �D@      C@     �I@     �F@     �G@     �F@     �B@      F@     �C@     �E@     �I@      L@     �L@     �A@      Q@     �I@     �G@      M@     �F@      K@     �E@     �K@     �K@      P@      Q@      P@     �L@      P@      N@     �N@      N@     �K@     @P@     @R@      R@     @P@      R@      Q@      M@     @R@     @P@      M@     �Q@     �M@     @P@      R@     �U@     @U@     @U@      V@     �V@     �U@      X@      T@     @T@     �V@     @Z@     �X@     �V@      X@     �V@     @R@      Z@     @Y@     �X@      W@      \@     @Z@     �W@      `@      `@     @]@     �[@     �^@     �\@     �[@     �[@     @_@      \@     �b@     �_@     �a@     �a@     �c@      \@     �`@     �b@     �b@     �h@     �g@     �e@     @b@     �c@     �p@     `f@     �e@     @d@     @e@     @e@      d@     `b@     �b@      e@      e@     �f@     �f@     �e@      g@     �f@     �g@      h@      h@     `l@      j@     `j@      l@      n@     `i@     �k@      n@      l@      k@     �o@      o@     �l@     @n@     �p@     @v@     �s@      o@     �p@     `s@     �s@      t@     0t@     Pr@     �u@     �t@     pt@      r@     pv@     0x@     Pv@     @w@     @{@     0�@     �x@     �y@     x@     0z@     �x@     �{@     �z@     py@     �{@     ~@     @     �@     ��@     ��@     ��@     �@     ��@     �@     ��@     8�@     X�@     ��@     p�@     ؈@     ؈@     `�@     ��@     ��@     ��@     Ќ@     h�@     ��@     (�@     �@     ��@     ��@     ��@     ܙ@     ��@     ĝ@     ,�@     ��@     �@     "�@     |�@     P�@     ��@     ��@     D�@     ��@     �@     |�@     ��@     0�@     B�@     s�@     �@     m�@     -�@     u�@     ��@     z�@     /�@     �@     k�@     ַ@     �@     ��@     Ǹ@     j�@     �@     ׼@     �@     G�@     �@     ļ@     Ѻ@     ��@     ��@     ��@     c�@     ְ@     ��@     �@     ��@     ��@     ʨ@     ܦ@     ��@     ��@     ��@     �@     ��@     ��@     &�@     ��@     �@     ʩ@     ��@     ��@     W�@     ��@     ��@     a�@     .�@     N�@     h�@     X�@     ��@     ��@     �|@     �y@     @s@     @n@     �@@      7@      &@        
�
predictions*�	   @�ۿ   � �@     ί@!  �0��@@).iK[C�T@2���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�>h�'��f�ʜ�7
�6�]���1��a˲���~��¾�[�=�k��
�/eq
�>;�"�q�>��>M|K�>�_�T�l�>pz�w�7�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�6v��@h�5�@!��v�@زv�5f@�������:�              �?              �?              �?      �?      @      @      $@      $@      4@      6@      ;@      9@      A@     �G@      H@     �F@      N@     �J@      E@      G@     �P@     �L@     @Q@     �R@     �Q@     @R@     @R@     �R@     @P@     @P@     @R@     �Q@      K@     �Q@     �H@      A@      B@      H@      B@     �D@      <@      E@      ?@      2@      9@      *@      &@      6@      (@      *@      4@      ,@      3@      "@      &@      "@      ,@      @      @      @      @      @       @       @      @      @      @      @       @      �?      �?       @      @       @              @      @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?               @      �?              �?              �?      �?       @      �?               @       @              �?      @      �?      @      @      @      @      @      @      @      @      @      @       @      "@      "@      "@      "@      "@      (@      &@      4@      .@      9@      8@      5@      :@      7@      <@      8@      ;@     �H@      D@      C@      F@     �G@     �L@     @P@     @Q@     �O@     �P@     @Q@     �R@     @R@     �Q@     �O@      K@      J@     �G@      C@      @@     �A@     �C@      C@      ?@      8@      2@     �A@      3@      9@      3@      6@      9@      5@      4@      1@       @      @      &@      0@      @      @      @      @      @      @      @      @      �?       @      @      @       @       @      @       @       @      �?      �?              �?              �?              �?        =��B4      8�� 	���}���A**�h

mean squared error|��<

	r-squared0E�>
�L
states*�L	   @q��   @ �@    �&A!H�l���@)�ֲ�&X�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@      $@      ,@      2@      <@     @V@     @[@     @d@     @n@     �v@     0�@     <�@     -�@     �@     ��@     |�@     ��@     �@     D�@     ��@     \�@     ��@     ��@     &�@     �@     �@     ��@     ��@     N�@     j�@     ҥ@     ަ@     p�@     �@     ��@     ��@     U�@     }�@     ��@     շ@     ��@     ��@     _�@     �@     ��@     ��@     b�@     g�@     ��@     ��@     ��@     ��@     �@     ҵ@     q�@     z�@     �@     @�@     X�@      �@     h�@     n�@     �@     y�@     x�@     s�@     �@     ��@     ޭ@     ��@     R�@     `�@     ȣ@     l�@     |�@     h�@     d�@     T�@     ��@     ��@     ��@     �@     ��@     L�@     t�@     <�@     0�@     �@     ��@     ��@     ��@     ��@     H�@     ��@     ��@     ��@      �@     �@     ��@     ��@     �@     ��@     h�@     Ђ@     �@     �~@     0�@     �~@     (�@     �|@     pz@     Py@     0y@     �x@     �z@      y@     `z@      x@     pv@     @v@     �v@      x@     �t@     �v@     Pv@     �u@     �v@     pu@     @t@     �s@     �u@     �r@     �s@     `r@     Pr@     0r@     �t@     pp@     p@     @p@     �o@     �q@     Pq@     �s@     �s@     �n@     �p@     `k@     �j@     `m@     @l@      g@     �i@      m@     �i@     �m@      g@      j@     �i@     �g@      h@     �h@     �e@     @f@     �d@     @f@      h@     @e@     �f@     �f@      i@     �e@     �f@     �c@      f@     `c@     �d@     �d@     `d@      d@     �c@     `d@      d@      a@     �b@      a@     @b@     �b@     `h@     �`@     �`@      g@      h@     @`@     �]@     �`@     @a@     @`@      `@     �`@      `@     �`@     @]@     �`@     ``@      b@     �X@      ^@      ^@     �_@     �\@     @Z@     �]@      Y@     �Y@     @Z@     �W@     @U@     �S@     �W@     �X@     �W@      V@     @Y@     �T@      U@     �R@     �Q@     �T@      V@      V@     @U@     @U@     �T@     �Z@     �g@     �S@      Z@     @S@     �P@     �P@     @R@      O@     �P@     �L@      O@     �P@      Q@      K@      R@     �P@     �M@      M@     �K@     �J@     @P@     �G@      P@     �L@     �N@     �L@     �F@     �H@     �K@     �P@     �T@      L@      H@     �F@      C@     �F@      A@     �F@      C@     �B@      F@      9@      H@      H@     �C@     �@@      ?@     �B@     �@@     �D@      @@      G@      >@      A@      B@      =@      @@      ?@     �H@     �@@     �B@      @@      =@     �C@     �A@      @@      9@     x�@      �@      6@      2@      <@      3@      4@      7@      4@      5@      ;@      8@     �A@      :@      A@      >@      5@      <@      4@      A@      B@      A@      ;@      C@     �@@      @@     �B@      =@      ?@      >@     �@@     �C@      B@      E@      ?@      E@      @@      @@      <@     �A@      G@     �A@      A@      A@     �@@     �C@      D@     �B@      H@      C@     �J@     �C@      H@     �J@      D@     �C@      H@      K@     �I@     �D@     �G@      M@     �J@      O@      H@     �L@     �G@     @R@      M@     @Q@     �N@      K@      K@      R@      F@     �Q@     �E@     �M@     @U@      T@     �O@     �P@     �T@     �P@      U@     �R@      V@      O@     �R@     @T@      U@      S@     @Z@      R@      X@     �T@      V@     �S@     �V@     �X@      [@     @W@     @T@     @Y@     �T@      ]@     �W@      Z@     �]@      ^@     �]@     �Y@     �\@     �Y@     @_@     @Z@     �\@     �]@      `@     @_@     ``@      a@     �`@     `b@     �a@     �a@      `@     �_@     �`@     �^@     @a@     �`@     �c@      g@     �p@     �i@     �b@     `b@     �e@     �a@     `d@     �c@     �e@     `b@     @g@     @f@     �g@     �f@     `f@      g@     `g@     @g@     �g@      h@     �i@     `k@     `i@      k@     �g@     �h@     @l@      n@     `l@      m@     �n@     �m@     �m@      o@     �v@     �v@     Ps@     `q@      r@     @q@      q@     @q@     `p@     `r@      t@     �t@     0t@     �u@      u@     `v@     �w@     {@     P@      y@     �x@     �z@     �w@     `x@      }@     �}@     �{@     �~@      ~@     P|@     0�@     �@     �@     �@     x�@     ��@     �@     �@     H�@     ؄@     X�@      �@     �@     ��@     Љ@     �@     ��@     X�@     ؋@     ��@     @�@     ��@     H�@     p�@     l�@     H�@     �@     X�@     ę@     �@     �@     j�@     D�@     ��@     ��@     l�@     r�@     Ƭ@     ��@     �@     в@     ��@     ȵ@     ��@     �@     f�@     L�@     ۹@     z�@     ��@     ޷@     ��@     D�@     ��@     k�@     �@      �@     �@     �@     ��@     �@     �@     ��@     ��@     e�@     �@     �@     �@     C�@     6�@     &�@     v�@     2�@     Z�@     �@     f�@     x�@     r�@     Z�@     ��@     f�@     ��@     ��@     t�@     ��@     :�@     ��@     ��@     ��@     m�@     �@     a�@     D�@     d�@     �@      �@     �@     �@     p�@      |@     �y@     @u@     �f@      ;@      ;@      @        
�
predictions*�	   ��E׿    I@     ί@!  �I�!@)ɬ"��KS@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��>h�'����(��澢f�����uE���⾮��%������>
�/eq
�>I��P=�>��Zr[v�>��[�?1��a˲?6�]��?����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@u�rʭ�@�DK��@�������:�              �?      �?      �?      �?      @      @      @      6@      :@      4@      :@     �@@      @@     �H@      C@      H@      N@     �Q@      N@     �T@      P@      T@     �S@      T@     �S@     @V@     @Y@     �Y@     @Q@     �V@     �Y@     @P@     @T@     �L@     �O@     �N@     �Q@      H@      I@      L@      F@     �E@      :@     �B@      5@      ?@      ?@      0@      8@      6@      "@      5@      .@      .@      &@      &@      *@      *@      @      .@      @      @      @      @      @      �?       @               @      �?              �?      �?      �?      �?              @               @               @      �?       @      �?              �?              �?              �?      @              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?      �?      �?       @              �?              �?      �?       @      �?      @      @      @       @      �?      @      "@      �?      @      �?      @      @      @      @      ,@      "@      @       @      .@      *@      "@      (@      1@      3@      6@      8@      3@      =@      8@      5@      8@     �@@      B@      A@     �F@     �G@      D@      ?@     �E@     �H@      E@      ?@     �F@     �@@      8@     �E@      ;@      F@     �B@      >@     �@@      7@      @@      8@      5@      8@      0@      4@      .@      2@      6@      2@      0@      .@      .@      5@      *@      (@      &@       @      @       @      @      &@      @      @       @      @      @              @      �?      @      @      @      @      �?       @              �?              �?              �?        �U24      e��	1~~���A+*�h

mean squared error��<

	r-squared��>
�L
states*�L	   �v��   ���@    �&A!��n�I<�@)���y7S�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              &@      *@      ,@     �A@     �J@      W@     �c@     @m@     ps@     �}@     �@     ē@     ��@     �@     ��@     ձ@     �@     �@     �@     *�@     �@     ��@     ��@     ̡@     �@     ��@     |�@     �@     ��@     ��@     �@     �@     \�@     ��@     ��@     ��@     �@     u�@     �@     e�@     ��@     p�@     ľ@     �@     ��@     ��@     Ҽ@     ��@     ��@     �@     i�@     ۷@     �@     ��@     ޶@     ��@     D�@     ¸@     ��@      �@     �@     ��@     �@     ��@     �@     _�@     ��@     ��@     x�@     �@     .�@     ��@     �@     V�@     �@     ��@     ��@     ��@     x�@     H�@     ��@     l�@     �@     L�@     ��@     �@     0�@      �@      �@     P�@     ��@     p�@     ��@     �@     �@     ��@     ��@     Є@     h�@     H�@     ��@     ��@     h�@     ��@     p@      �@     ��@     �~@     �{@     p}@     p{@     �y@      y@     y@      x@     y@     {@     �y@     @x@      {@     Pw@     @w@     �y@     �u@     �t@     �u@     �w@     �t@     �r@     �r@     0t@     �s@     Pr@     @s@     �r@     s@     �p@     @o@     `q@     �n@     �l@     �q@     @t@     0q@     @o@     �n@      m@      o@     `m@     `i@     �m@     �k@      k@     �h@     �i@     �h@      l@     `i@     �i@     �f@     �h@     `g@     @g@     �d@     `g@      i@     �g@     �f@     `d@     �f@     �d@      e@     �g@     �b@     `e@     �f@     �c@     `e@     �b@      c@     �b@      b@      e@      f@     ``@     �a@     �e@     �`@     �c@     `e@     �_@     �h@      i@     `c@     �d@     �b@     @`@     @^@     `a@      _@     ``@     �`@     �`@      a@      `@     �`@     �_@      Z@     �`@     @`@     �\@      Z@     �V@     �Z@      Y@     �[@     �Y@     �Y@     �V@     �T@      X@      X@     @W@      V@     �U@     �V@     �Y@      Y@     @V@     �W@      V@     @P@     �U@     �V@     �R@     @W@      R@     @R@     �Q@     �R@     @Q@     @Q@     �K@      O@     @P@      R@     ``@     �T@      M@      K@     �Q@     @Q@      I@     �O@     �N@     �H@     �N@     �J@     �I@      F@     @S@     �J@      L@      K@      F@      K@     �F@     �E@      I@      N@      P@      K@     �F@      G@     �J@     �C@     �D@      G@     �E@     �A@     �B@      >@      E@     �C@     �E@      C@      B@      D@      @@     �G@      G@     �@@     �@@      ?@      A@      4@      =@      :@      B@      ;@      F@      2@     �@     ��@      3@      9@      :@      $@      5@      0@      :@      :@      8@      8@      9@      ?@      >@      @@      ?@     �@@      0@      A@      7@      B@      @@      C@      7@     �C@      F@      A@      @@     �A@      C@      B@      ?@     �A@     �@@      D@     �F@      8@      C@     �D@     �F@     �A@     �D@      >@      B@     �E@     �E@      D@     �N@      D@      K@      B@     �F@      >@      >@      K@     �E@     �I@     �F@      D@      J@      L@     �M@     �P@      R@      O@     �Q@     �M@      J@      K@     �G@     �K@      M@     �L@     �S@     �P@     �Q@     �R@      O@      P@     �Q@     �J@      O@     �Q@     @U@     @V@      V@      T@      V@     �U@     �Q@     �U@      X@      T@     �Y@     �W@      V@     @Z@     @X@     �U@     �U@     �[@     �Y@     �X@     �Y@     @Y@     �Y@     �Y@     �V@     �Y@     �\@     @]@     �^@      \@      ]@     @`@     �\@     @a@     �^@      ]@     �g@      d@      b@     ``@     ``@     �]@     �a@      d@     �a@     �a@     �a@     @b@      a@     �b@     �f@      n@     �e@     `b@      f@      d@     �c@      d@      g@     �g@      e@     @e@     �h@     �g@      i@     �d@      g@     `f@     �k@     `j@      k@     �k@     @j@      j@     �j@     �k@     �i@     @l@     �p@     0p@     �m@      m@     �n@     ps@     0u@     �v@     @t@     0s@     Pt@     �p@     Pq@     0v@     �t@     `t@     �u@     �s@     �t@     �v@     �s@     `x@     �u@     �v@     �x@     ��@     Pz@     �z@     Pz@     �{@     z@     �}@     �}@     �}@     �|@      ~@     @     �@     ��@     �@     ��@      �@     ��@     ��@     0�@     ��@      �@     p�@     ��@     @�@     ��@     ��@     0�@     �@     ��@     8�@     �@     ܑ@     ��@     4�@     �@     �@      �@     T�@     �@     �@     0�@     l�@     ��@     >�@     ��@     d�@     r�@      �@     �@     �@     ��@     ��@     :�@     `�@     j�@     9�@     ��@     ӹ@     ��@     ɸ@     �@     @�@     ��@     �@     +�@     H�@     )�@     ��@     ��@     j�@     N�@     |�@     �@     Խ@     (�@     ��@     %�@     ��@     '�@     ��@     5�@     �@     �@     ��@     ��@     �@     ��@     ��@     j�@     `�@     J�@     v�@     ޤ@     "�@     :�@     �@     f�@     ��@     �@     �@     k�@     s�@     ��@    �K�@     ,�@     ��@     ��@     ��@     Ȁ@     `|@     P{@     `v@     @W@      >@      8@      $@        
�
predictions*�	    tv�   �}@     ί@!  ���T9@)�m�qU@2�uo�p�+Se*8俰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x�������6�]���>�?�s���O�ʗ���})�l a��ߊ4F��iD*L��>E��a�W�>})�l a�>pz�w�7�>>�?�s��>�FF�G ?x?�x�?��d�r?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?h�5�@�Š)U	@{2�.��@!��v�@�������:�              �?              �?              �?              �?      �?      @      @      @      "@      0@      5@      6@      <@      B@      A@      G@     �G@     �K@      N@      S@     �O@     �R@     @U@     �P@      P@     �R@     �S@     �S@     �P@     @X@     @Q@     �P@     @S@     �P@     �P@     �Q@      O@     �G@     �F@      C@     �D@      D@      @@     �C@      =@      8@      7@      7@      6@      .@      .@      .@      1@       @      2@      *@       @      $@      @      @      "@      @      @      @               @      @      @      @      @      @              @       @              @      @              �?       @      �?      �?               @              �?      �?      �?               @              �?              �?               @              �?              �?              �?              �?              �?               @      �?              �?              @              �?      @       @      @      �?       @      �?      @      @       @      "@      @       @      $@      @       @      @      @      @      @      $@      0@      $@      &@      (@      8@      .@      5@      <@      ?@      <@      ;@      E@      ;@      ?@      :@      A@     �C@      I@      G@      I@      I@     �P@      C@     �F@      C@      P@      G@     �E@     �D@      F@      C@      D@      A@      >@      =@      ?@     �A@     �C@      9@      4@      <@      8@      4@      ?@      =@      *@      (@      ,@      0@      4@      @       @       @       @      @      @      $@      @      @      �?      �?       @      @      �?      �?      @      @      @      �?              �?              �?              �?        �!�U4      e(��	7�~���A,*�h

mean squared error8�<

	r-squared�d�>
�L
states*�L	   ����   ` �@    �&A!���
��@)��U�	��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@      0@      9@     �B@     @P@     �Z@     �i@     �p@      w@     ��@     P�@     ��@     }�@     ��@     �@     c�@     4�@     ��@     ҥ@     p�@     ��@     �@     ��@     &�@     &�@     :�@     L�@     :�@     ��@     N�@     L�@     D�@     ��@     p�@     ڭ@     ��@     �@     �@     ��@     �@     �@     j�@     ٽ@     �@     U�@     ��@     5�@     ��@     n�@     ��@     ��@     �@     u�@     �@     ��@     3�@     ��@     ?�@     �@     ��@     ��@     ��@     ٶ@     ��@     ʳ@     _�@     }�@     �@     t�@     ��@     $�@     $�@     �@     �@     h�@     4�@     ��@     ě@     0�@     L�@     D�@     X�@     ��@     �@     �@     ��@     ��@     ��@     Ȍ@     0�@     Љ@     ��@     ��@     ؈@     H�@     �@     X�@     ��@     ��@     ��@     ��@     �@     X�@     ��@     ��@     8�@     �|@     x�@     �{@     �|@      |@     �z@     |@     �x@     py@     �z@     �y@      y@     �x@     �y@     �v@     0t@      t@     `u@     �v@     �t@     `u@      u@     �s@      v@     �r@     t@     ps@     r@      q@     0r@     �s@     �s@     �q@      r@     Pq@     p@     p@     �l@      n@     �p@     �l@     �k@     �j@     �m@     @l@      h@     @j@     @i@     �h@     �k@     �i@     �j@     �g@      i@     @i@     @e@     @j@     @f@     �f@     �f@      f@      g@     �d@     �i@     `f@     `f@     �f@      d@     �c@     �e@     �c@     `c@     `d@      d@     �a@      b@     �c@     �a@      d@      e@     �`@     �a@     `b@     �_@      ^@     �d@     �i@     �_@     �_@     �_@      \@     �]@     �b@     �a@      `@     �]@      _@     ``@     �b@     �_@      `@     @^@     @`@     �]@     @\@     �b@      [@      `@      V@     �W@     @]@      ^@     �[@     �V@      Y@     �X@     �\@     �W@     �X@      W@     �W@     @V@      T@     @R@     �Q@     �U@     �T@      S@     @T@     �T@      U@     �J@      Q@      Z@     @d@     @V@      Q@     @P@     �O@      O@      N@      P@     �M@      I@     �P@     �J@      P@      G@     �L@     �K@     �L@     �J@     @P@     �K@     �H@     �L@      F@     �H@      J@      K@      I@     �O@     �J@      E@      D@      O@     �E@     �N@     �K@      H@     �D@      D@      F@     �D@      @@      F@      A@     �B@      B@      ?@      9@      <@     �F@     �C@      ;@      ?@      =@      B@      8@      9@      @@     �A@      <@      :@      4@      >@     ܝ@     �@      4@      8@      8@      <@      5@      <@      .@      1@      @@      @@      :@      >@      >@      =@      <@      :@      4@      @@      8@      <@      =@      ?@      :@     �B@     �B@      A@     �D@     �@@      ;@      C@      A@      ?@     �C@     �E@     �@@      D@      ?@      @@     �G@     �E@     �A@     �E@      H@      G@      E@     �@@     �F@      F@      F@     �D@      F@     �K@     @P@      L@      J@     �J@      J@     �I@     �K@     @P@     �K@      I@     �H@     �J@      G@      K@     �O@      M@     @P@     @R@     �P@     �P@      L@      I@     �J@      L@     �Q@     �L@     �O@     @P@      S@     �S@      O@     @U@     @W@     �T@     �U@     �W@      W@      U@     @W@     �U@     �V@     �T@     �V@      Z@     �V@     @W@     �S@      T@     @X@     @U@      ^@     @W@     �Y@      X@     �[@     �X@      [@     @_@      [@     @]@      \@     �`@     ``@      k@      a@      a@     �a@      ]@     �`@     �_@     �`@     @^@     �`@     �a@     `c@     �a@     @c@     `a@     `b@     `e@     �b@     �k@     `k@      g@     `c@     �f@     �d@     �c@      c@     �f@     �g@     �e@     �h@     `j@     @e@     `f@     @i@     �g@     @g@     �i@      i@      f@     `n@     �h@     �j@      m@      l@     �m@     �n@     `n@      k@     �m@     p@     �s@     @t@     �r@     �p@      r@     �q@     �s@      r@     0u@     @w@     �t@     �s@     �v@     �w@     �u@     pu@      v@     v@     �w@     �v@     �x@     ��@     �y@      y@     0|@     {@     �{@     {@     �{@      }@     �@     X�@     Ȁ@     0@     �@     ��@     ��@     �@     P�@     (�@     Ȇ@     X�@     `�@     �@     ��@     H�@     x�@     p�@     (�@     p�@     H�@     �@     ��@     ��@     Ĕ@     @�@     |�@     P�@     ��@     ܛ@     P�@     <�@     ̡@     ʢ@     .�@     ��@     ��@     ^�@     b�@     ��@     �@     ��@     �@     E�@     ��@     �@     ��@     W�@     ʹ@     �@     h�@     �@     6�@      �@     u�@     ��@     /�@     6�@     <�@     ��@     Ѻ@     =�@     m�@     2�@     �@     �@     �@     V�@     v�@     S�@     S�@     X�@     ��@     8�@     ~�@     ��@     ҩ@     H�@     >�@     ��@     �@     r�@     *�@     Υ@     �@     ĥ@     ��@     ��@     ��@     (�@     ��@     *�@     ��@     ֱ@     q�@     j�@     0�@     p�@     P�@     ��@     `}@     P@     y@      K@      :@      :@      @        
�
predictions*�	   �f~�   `�[@     ί@!  @b���)O��^�*Q@2�+Se*8�\l�9⿰1%�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���5�i}1���d�r�1��a˲���[�����%ᾙѩ�-߾��~]�[�>��>M|K�>1��a˲?6�]��?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@u�rʭ�@�DK��@�������:�              �?      �?              �?               @      @      @      @      &@      .@      7@      4@     �A@      =@      A@     �D@     �N@     �M@     �Q@     �P@     �R@     �U@     @S@     �P@     �P@     @R@     @T@     @W@     �U@     @T@     �R@     @R@     �S@     @Q@      R@     �M@     @P@      H@     @P@      P@      H@     �C@      C@      9@     �B@      7@      @@      7@      1@      =@      4@      2@      ,@      4@      (@      &@      2@      *@       @      @      "@      @      @      @      @      @      @      @      @      �?      @      @      @               @       @      �?      �?               @      �?       @       @              �?       @              �?               @              �?              �?              �?              �?              �?              �?       @               @              �?              �?      �?              �?       @       @              �?      �?      @      �?      @       @      @      @      @       @      @       @      @      @      @      @      $@      @      $@      &@      "@      0@      (@      &@      ,@      1@      0@      5@      6@      <@      =@      <@      <@      8@     �E@      <@     �B@     �A@     �B@      D@      D@     �F@     �F@      G@     �G@     �F@     �G@      F@     �C@      C@     �F@     �A@      D@      ;@     �A@      =@      7@      ;@     �B@      3@      :@      3@      <@      :@      "@      *@      ,@      *@      *@      &@      (@      "@       @       @      @      @       @       @      @      @       @       @       @      @       @      �?      �?               @              �?              �?              �?        ��I R4      d��	�`%~���A-*�h

mean squared errorv��<

	r-squared�C�>
�L
states*�L	   ���   ��@    �&A!�q���5�@) P2���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@      0@      (@      5@     �G@      V@     �h@     �h@     �x@     p�@      �@      �@     e�@     "�@     l�@     ��@     <�@     @�@     r�@     <�@     v�@     J�@     .�@     B�@     ��@     �@     �@     ֡@     Ȣ@     ��@     �@     Х@     ��@     ��@     d�@     ,�@     �@     ��@     ��@     �@     %�@     $�@     ��@     �@     ſ@     ��@     c�@     �@     �@     �@     R�@     ��@     (�@     h�@     �@     �@     a�@     и@     0�@     ��@     P�@     �@     ��@     �@     I�@     j�@     ��@     �@     ��@     |�@     B�@     `�@     l�@     ��@     ��@     ��@     \�@     �@     \�@     d�@     (�@     p�@     ��@     T�@     @�@     Ȑ@     X�@     �@     �@     �@     �@     ��@     ��@     8�@     x�@     ��@      �@     ��@     X�@     Ђ@     ��@     ��@      �@     �~@     ��@     (�@     z@     �{@     0}@     P|@     pz@     `y@     @{@     P{@     0|@     �z@      y@     �v@     @v@     �w@     `v@     �u@     `v@     �t@     Pt@     @t@     �w@     �s@     �r@     �s@     `r@     �t@     0r@     pq@     �s@     `s@     0r@     �r@     �r@     �t@      q@     �n@     Pp@      n@      o@     �o@     �l@     `l@      l@     �j@      f@     �g@      i@     �l@     @g@      j@     �g@     �g@     �j@      g@     @j@      d@     �g@     `i@      i@     �h@     �c@     �d@     �d@     �f@     �d@     �e@     �d@     `b@     �e@      f@     @`@     �_@      c@     `c@     �c@     �c@     @f@      `@     `a@     �c@     �b@     �_@     ``@     �_@     �^@      j@     �h@     @^@     �_@     �`@      a@      _@     �Z@     �\@      ^@      `@     �`@     �b@     ``@     @_@      _@      X@     @\@     �[@      _@     �W@     @a@      Y@     �X@     @\@     @`@     �[@     �X@      X@     �_@      W@     @X@      X@     �Y@     �X@     @W@     �U@     �X@     �U@     �W@     @U@     �V@     @S@      U@     @Q@     @R@     �d@     �X@      Q@     @P@      K@      R@     �P@     �R@     �T@      L@      M@      L@     @R@      G@     �K@     �J@      I@     �P@     �H@     �L@     �J@     �M@     �P@     �I@      I@      L@      F@      H@     �J@     �G@     �C@     �I@     �J@      A@     �H@     �K@     �H@      K@      I@     �J@      N@      A@      A@     �C@      D@      D@      B@     �C@     �C@      A@     �C@     �B@     �E@      :@      ?@      D@     �C@      ;@      >@     �B@      ;@      4@      9@      @@      :@     ��@     �@      @@      9@      8@      =@      8@      1@      9@      8@     �C@      B@      =@      8@      =@      9@     �B@      ;@      :@      ;@      A@      =@      A@     �B@      A@      6@     �@@     �A@      =@      C@     �B@      C@      >@     �A@      F@      =@      I@     �D@     �B@      >@     �B@     �B@      6@      F@      D@     �G@     �E@      ?@     �D@     �D@      C@     �J@      K@     �I@      I@      C@      G@      P@     �K@     �E@      I@      L@     �H@      N@      O@     �G@     �N@      G@     �P@     �P@      P@      J@     �Q@     �R@      N@     @Q@     �L@      Q@     �R@      Q@     �Q@      L@     �Q@     �R@     �S@     �Q@      V@      O@     �R@     @P@     �U@     �O@     �S@     �S@     �Q@     �V@      X@     �U@     �Y@     �V@      Y@     �Y@     �Y@     @X@     @[@      W@     �Y@     �X@      Z@     �X@     �Z@     �\@     �\@     �U@     �_@     �\@      X@     �f@     �d@     @a@     �_@     �]@     @]@     �a@     �`@     �b@     �c@     �a@     @d@     @b@     �c@     �a@      c@     �b@     @d@     �k@     �h@      h@     �c@     �a@     �e@     �c@     �f@     �e@      e@     `g@      h@      c@     �h@     `i@     �i@      i@     `e@     �h@     �i@     �i@     �i@     @k@     @i@     �h@     �g@     �m@      o@      n@     �j@      m@     Pp@     �s@     �s@     0q@     �r@     0q@     �p@     Ps@     0t@     �r@     `r@     0t@     �w@     pu@     �u@     u@     �t@     pt@     u@     �u@     �w@     @y@     x�@     �@     �x@     x@     �y@     {@     0|@     ~@      |@     �~@     @~@     `�@     �@     h�@      �@      �@     0�@     ��@     ؃@     Ѕ@      �@     8�@     0�@     Ȉ@     ��@     `�@     ��@     Ќ@     ��@     ��@     8�@     ̑@     ܒ@     ��@     ��@     ��@     4�@     �@     ��@     ��@     N�@     �@     z�@     �@     ~�@     L�@     �@     ��@     ��@     M�@     [�@     ��@     b�@     ��@     w�@     Z�@     �@     �@     "�@     ��@     �@     ��@     �@     ��@     N�@     ��@     ��@     S�@     ��@     ��@     n�@     �@     �@     �@     �@     �@      �@     E�@     I�@     ��@     ��@     �@     L�@     ��@     Ī@     ��@     ��@     N�@     l�@     ��@     ��@     f�@     ��@     �@     L�@     "�@     X�@     ��@     v�@     ñ@     ��@     ��@     U�@     ��@     £@     ��@     �@     Ȇ@     @�@      ~@     �@     `{@      G@      =@      2@      @        
�
predictions*�	   ��E�   ��r@     ί@!  /�cD@)��9 dU@2�uo�p�+Se*8俰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7����5�i}1�x?�x��>h�'��>�?�s���O�ʗ���pz�w�7��})�l a򾮙�%ᾙѩ�-߾})�l a�>pz�w�7�>I��P=�>��Zr[v�>��[�?1��a˲?>h�'�?x?�x�?��d�r?�5�i}1?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�DK��@{2�.��@�������:�              �?              �?              �?       @      @       @       @              @      @       @      (@      (@      2@      8@      <@      <@     �@@      F@     @R@      M@      J@     @Q@     �Q@     @W@      U@     �U@     @R@     @S@     �U@     �Y@      T@     �U@     �T@      R@     �P@      O@      S@     @Q@     �Q@     �I@      J@     �J@     �D@      E@      :@      ;@      ;@      4@      1@      2@      :@      *@      4@      @      $@      &@      (@      "@      @       @      @      @      @      @      �?       @      @       @      @      �?      �?      @      �?      �?      �?      @               @       @              �?      �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?      �?               @              �?      �?       @       @      @      �?       @      @       @      @      @       @      @      @      @      @      @      "@       @      $@      @      $@      (@       @      &@      (@      "@      $@      9@      2@      ;@      9@      :@      :@      7@      =@      ;@     �@@      @@     �D@      G@     �M@     �H@      G@     �D@     �N@      D@     �F@      G@      J@      B@     �E@      E@     �@@      =@     �@@     �A@      ?@      <@      ?@      >@      5@      2@      7@      6@      6@      ;@      *@      5@      &@      0@      8@      .@      &@      $@      &@      "@      $@      @      @      @      @      @      @       @      �?      @      @      @      �?               @      �?      �?              �?              �?        ���3      �3�	%9~���A.*�g

mean squared errort)�<

	r-squared���>
�L
states*�L	   ����   �!�@    �&A!�Y����@)�@"U�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              "@      .@      "@      5@      B@     @]@     �l@     `n@     `u@     @�@     ��@     �@     ��@     r�@     ��@     ��@     �@     ��@     ȥ@     �@     ʣ@     ��@     �@     z�@     H�@     ��@     ��@     ��@     &�@     ��@     \�@     ��@     `�@     d�@     N�@     "�@     |�@     �@     ��@     {�@     <�@     T�@     >�@     ��@     Z�@     a�@     �@     j�@     ��@     h�@     -�@      �@     �@     8�@     b�@     V�@     X�@     ��@     ��@     ��@     .�@     x�@     ʷ@     %�@     �@     ��@     ��@     ��@     �@     z�@     >�@     ��@     �@     С@     �@     @�@     �@     �@     @�@     �@     d�@     ��@     Г@     �@     ,�@     �@     8�@     h�@     ��@     ��@     ȉ@     ��@      �@     ��@     ��@     P�@     ��@     ��@     ��@     (�@     Ȅ@     ��@     H�@     Ȁ@     �|@     �}@     P@     ~@      |@     `|@     Py@     �z@     �y@     �w@     �y@     �y@     �x@     y@     �v@     `v@      w@     v@      w@     �u@      w@     @u@      t@     0r@     �s@     �t@      s@     �p@     @q@     �r@      q@     �q@     Pt@     �t@     �q@     @p@     �p@     `k@     �o@     �p@     �k@      m@     `k@     `l@     �m@     �k@     �j@     `i@     �f@     �f@      g@     @k@     �k@      h@      g@      g@      j@     �f@     �f@     �d@      g@     `g@     �b@      i@     �e@     @d@     �c@     `c@     `a@     �e@     �d@     �`@     `e@     �g@     `b@      c@      d@      c@     �`@     �_@     @c@     @c@     ``@      `@      a@      ]@     �b@     �j@      j@     �^@      `@     �[@      ]@     �a@     �`@     �[@     @a@     �\@     �b@     `b@     ``@     �]@     �Z@     @^@     �[@     �W@     �\@     @Z@      \@     �\@     @Y@      Y@      Z@     �[@     �Z@      ^@     @[@     @U@     �\@     �Z@     �W@     @X@     �V@     �P@     @V@     �W@     �T@     �U@     @\@     @Y@     �T@      R@     @U@      d@      `@      Q@     �Q@      N@      U@     �Q@     @R@     �H@     �L@     @P@     �N@     �O@     �P@     �L@      R@     �J@     �I@      O@     �I@     �B@      P@     �N@      J@      E@      I@      D@      D@     �H@      K@     �G@     �H@     �E@     �G@     �D@      G@     �K@      C@     �H@     �G@     �L@     @P@     �A@     �C@      ?@     �A@      B@      C@     �@@      =@      I@     �D@      >@      @@      ?@      B@     �A@      >@      8@     �B@     �@@     �C@      ;@      8@      >@     8�@     ��@      :@      :@      1@      5@      2@      6@      A@      :@      ;@     �A@      ?@      5@      6@      8@      :@      >@     �F@     �@@      ?@      C@      <@      >@      H@     �B@      H@      9@     �C@      C@     �A@      >@      =@      C@     �B@      @@     �A@     �B@      G@     �C@     �G@      E@     �A@      @@     �G@      7@      ?@     �E@     �F@     �G@     �L@      A@     �J@      M@     �J@     �G@     �D@     �H@     �I@     �M@     �L@      C@      L@      O@     �Q@     �K@      Q@     �C@     �F@     �O@     �J@     @Q@     �S@     @R@      H@      O@      Q@      F@     @Q@     �P@      O@     �Q@     @Q@     @Q@     @R@     �P@     �P@      V@     @R@     @U@     �S@      Q@     �R@      T@      R@      S@      V@     �W@     �V@     �T@     �S@     @[@     �W@     �W@     @V@     @Y@     @Y@      _@      [@      \@     �Z@     �`@     @[@      [@      [@     @^@      `@     �^@     �^@      Y@     @`@      ^@     @a@     `a@     @k@     �d@     �`@     @`@     �`@     �c@      a@      _@     �`@     �b@     @f@     `e@     �p@     `f@     �g@      d@      d@     �e@     �f@     �e@     �e@     @f@     �d@     `h@     �h@      h@     `h@     �g@     `e@     `i@     @i@     @h@     �j@     �i@     �j@     @i@     �i@     @i@     �n@     @o@     �p@     �n@     q@     @o@      p@     Pp@     �r@     �s@      u@     �r@     `s@     @r@      r@     �t@     �s@     �s@     `t@     �u@     �v@     �x@     �|@     Pw@     �z@      z@     `}@     �~@     �z@      {@     �z@     �{@      }@     �{@     �|@     �~@     �~@     ��@     p�@     Ȁ@     ��@     h�@     �@     x�@     X�@     `�@     @�@     �@     8�@     H�@     ��@     ��@     ��@     ��@     ��@     P�@     ��@     ܑ@     t�@     L�@     Ԕ@     ؕ@     p�@     ̘@     ��@     ,�@     t�@     
�@     F�@     ܣ@     j�@     ��@     ��@     (�@     ٰ@     ܱ@     [�@     ��@     �@     ��@     ��@     ��@     O�@     ҹ@     Ѹ@     ��@     E�@     +�@     �@     6�@      �@     [�@     ��@     a�@     ׹@     ��@     ��@     �@     O�@     ��@     �@     �@     ��@     ��@     s�@     g�@     ұ@     v�@     �@     ��@     �@     r�@     :�@     ,�@     L�@     :�@     ~�@     �@     �@     (�@     Х@     �@     ,�@     B�@     �@     ñ@     u�@     d�@     ��@    �o�@     ��@     ��@     (�@     0�@     ��@     H�@     0�@     �}@      _@      A@      >@       @        
�
predictions*�	   ��}�   @"�@     ί@!  ��_"@)$hUtU@2�uo�p�+Se*8�W�i�bۿ�^��h�ؿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������1��a˲���[��E��a�W�>�ѩ�-�>x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@{2�.��@!��v�@�������:�               @              @               @      @      @      @      $@      (@      ,@      4@      @@      4@      D@     �D@     �F@      N@     �P@     �L@     �Q@     �W@      S@     �V@     �W@      V@     �W@      [@      [@     �T@     �X@     �V@     �O@     �T@     @V@     �S@      M@     �P@      N@      J@     �E@     �F@      E@     �D@     �B@      A@      5@      ,@      :@      *@      :@      ,@      3@      (@      @      *@      @      *@      @      @       @       @      @      @      @      @      @       @      @       @       @      �?      �?              �?      @              @              �?              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @              �?      @      �?      @      �?      @              �?      �?      @      �?      @      @       @      @              @      �?      @       @      @      @      @      &@      $@      $@      "@      0@      1@      2@      4@      8@      *@      7@      8@      @@      7@      9@      @@      :@      <@      B@      F@     �B@     �I@     �F@      I@      G@     �F@      F@      D@      E@      :@      >@      @@      ;@      ;@      7@      9@      2@      9@      ?@      4@      7@      5@      2@      .@      5@      3@      *@      .@      @      (@      &@      0@      (@      "@       @       @      @      @       @      @      @      @       @      �?      @      �?      �?      �?               @      �?              �?      �?      �?              �?        ��8"4      9��	C�J~���A/*�h

mean squared errorN��<

	r-squared*3�>
�L
states*�L	    r��   �E�@    �&A!;*���@)\��(��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              $@      ,@      C@     �C@      I@     �c@     �p@     `s@     �x@     ��@      �@     ��@    ��@     ��@     P�@     �@     .�@     .�@     ��@     ��@     ~�@     ��@     �@     ̣@     p�@     ��@     �@     ��@     ��@     :�@     ~�@     ��@     �@     V�@     �@     �@     ��@     �@     Ҵ@     ��@     ��@     �@     ��@     ��@     ,�@     ��@     ��@     ��@     5�@     (�@     }�@     ȷ@     ��@     ˶@     �@     K�@     ?�@     ;�@     �@     ��@     ��@     +�@     ��@     z�@     ��@     (�@     ��@     #�@     �@     ~�@     ��@     "�@     �@     ̣@     ��@     ��@     L�@     ��@     x�@     �@     ��@     ��@     x�@     ��@     ��@     0�@     p�@     ؏@     ��@     p�@      �@     ؈@     ��@     8�@     ��@     ��@     �@     p�@     p�@     ȃ@     8�@      �@     �@     Ё@     H�@     �~@     �~@     �}@     �|@     �{@     �z@     �x@     `z@     `{@     �x@     �x@     `x@      x@     0v@     �w@     �v@     �u@     �w@      v@     t@     v@     �t@     `t@     u@     �q@     �r@     r@     Pp@     �q@      q@     �t@     s@     �q@     `q@     �m@     �n@     �o@     �n@     �k@     @l@     `k@     `j@     @n@      k@     �j@     �f@      k@      i@     `i@     �i@     �g@     �i@     �e@     @j@      l@     @f@     `g@      f@     �d@     `e@     �c@     �e@     �d@     @e@     �c@     �d@     @f@      a@     �d@     `e@     �c@     �a@     �e@     �b@     �`@     �_@     �a@      b@     �`@     �b@     �_@     @`@     �`@     �b@     �`@     @c@     `n@      c@     �\@     �_@     �\@     �_@     �^@      `@      `@     @^@     @\@      [@     `b@      d@     �a@      `@     �\@     �X@      ^@     �Z@     �]@     �^@     @Z@      Z@     �R@     @^@      S@      [@     @Z@     �\@     �_@     @Z@     �V@     �V@     �U@     �R@      S@     �T@      W@     �U@      Q@     �T@      R@     �R@      R@     @R@      U@     @f@     @Z@     �Q@      R@     �M@      T@     �U@     �S@     �R@     �O@     �R@      O@     �H@     �M@     @P@     @P@     �N@     �I@      T@      F@     �L@     �I@      M@     �N@      G@      G@      K@      J@     �E@      D@     �J@      C@      J@      G@     �H@     �G@      J@     �F@      H@     �D@      I@     �K@      O@      K@     �F@     �@@      @@     �A@     �C@      9@      B@      D@      @@      8@      A@     �@@      3@     �F@     �B@     �A@      =@      A@     �@@      6@     М@     D�@      3@      &@      6@      1@      :@      7@      =@      8@      7@      :@      ?@     �@@      6@      6@     �F@      ;@      :@      ;@      2@      6@     �@@      9@     �A@      ;@      B@      >@      F@      ;@      ;@      ?@      D@      ;@      B@      9@      I@      C@      A@     �A@      A@      >@     �D@      H@      ;@      L@      B@      D@      I@     �B@     �E@      K@     �C@      G@     �I@     �E@      G@     �I@      P@      H@      N@      K@      N@     �J@      L@     �I@      K@     �K@      K@     �H@     �K@      O@     �H@     �P@     @Q@     �J@      E@     �Q@     �Q@     @P@      J@      Q@     �P@      Q@     �S@     �P@     �U@     �O@      R@     @S@     �U@     @Z@     �Q@      Q@     �P@      T@     �V@     �W@     �V@     @Q@     @Z@     �U@     @V@     �U@     �S@      Y@     �W@      X@      [@      Z@     �]@      W@     ``@     �]@     �Y@     @Y@     �]@      ^@      ^@     �Z@      ]@      _@     �`@     ``@     �`@     �`@     ``@     �a@     �a@     �b@     @^@     �`@     �a@     �a@     �i@     �h@     �g@     �c@     `l@      m@     �d@     �d@     �d@      e@      g@      f@      g@      f@     @g@      f@     �i@      h@     �g@     �i@     `f@     @g@     �i@     �j@      k@      h@      i@     �n@     `p@     �n@     �k@      n@      o@     �o@     �o@     `o@     �r@     0p@     �p@     `t@     �w@     �u@     0u@      x@     �u@     �s@     �s@     Pu@     t@     0u@     pv@     0x@     Px@     �x@     �y@     �x@      �@     �|@      z@     �|@     �{@     @|@     p�@     0�@     `�@     ��@     `}@      �@     X�@     Ȃ@     x�@     X�@      �@     ��@     ؅@     ��@     (�@     p�@     ��@     �@     ؋@     ��@     H�@     ��@     �@     4�@     �@     p�@     ��@     \�@     �@     x�@     �@     ��@     ��@     r�@     �@     ��@     �@     *�@     ��@     T�@     (�@     
�@     �@     ô@     8�@     �@     b�@     ��@     �@     x�@     �@     ��@     ��@     ��@     d�@     ,�@     �@     �@     ^�@     %�@     ˹@     �@     �@     :�@     ��@     G�@     ��@     ֻ@     ��@     =�@     ��@     C�@     ��@     0�@     Z�@     6�@     v�@     ��@     ��@     ¥@     �@     �@     �@     ��@     ��@     ��@     ��@     �@     v�@     έ@     ��@     ��@     �@     ��@     ��@    ���@     �@     (�@     Ј@     ��@     P�@     �~@     �@     @{@     �h@     ``@      9@      "@        
�
predictions*�	    X+߿    y@     ί@!  �1T�@)��N�mS@2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�f�ʜ�7
������E��a�W�>�ѩ�-�>��Zr[v�>O�ʗ��>>�?�s��>��[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@{2�.��@!��v�@�������:�              �?      �?       @              @       @      @      @      @      1@      .@      :@      :@      C@      A@      K@     �G@     �D@     �P@     �L@     �O@      T@     �P@     �V@     �W@     �W@      R@      U@      V@      U@     @S@     @Q@     �P@     �Q@     @S@      H@      Q@      J@      K@      G@      J@     �C@      9@     �A@      ;@      B@      ;@      7@      9@      1@      7@      1@      *@       @      (@      "@      (@      @      @      @       @      @      "@      "@       @       @      @       @      @       @       @      @              �?       @      �?      �?       @      �?              �?              �?              �?      �?      �?       @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?               @      �?       @       @      @       @      �?       @      @      �?      @              �?       @      @       @      @      @      @      $@      @      @      @      @       @      $@      ,@      $@      (@      &@      &@      3@      7@      "@      ,@      5@      7@      5@      ;@      @@      A@      A@      C@      C@     �D@     �J@      B@     �L@     �F@     �H@     �K@      E@      H@     �F@     �J@      E@      B@     �D@      A@      @@     �A@      B@      C@     �C@      :@      :@      6@      ,@      9@      6@      1@      3@      0@      6@      0@      2@      ,@      "@       @      @      @      @       @       @       @       @      @       @               @       @      @      �?      �?      �?       @              �?              �?              �?        ��-�3      EN�	�\~���A0*�g

mean squared error5S�<

	r-squared�~�>
�L
states*�L	   @���    �@    �&A!qF����@)�{�����@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              &@      A@      J@      C@     @V@      d@     Pt@     �v@     0|@     0�@     ��@     ��@    �S�@     ��@     ��@     ��@     ��@     ��@     �@     ��@     |�@     h�@     ~�@     f�@     �@     ��@     ��@     �@     h�@     �@     ,�@     �@     �@     b�@     �@     ��@     t�@     x�@     �@     �@     ��@     i�@     l�@     Ӿ@     ��@     $�@     Z�@     �@     ?�@     ��@     ˷@     y�@     0�@     �@     O�@     ��@     ��@     0�@     ��@     �@     |�@     ��@     ��@     �@     ��@     �@     ��@     ��@     ��@     h�@     ��@     ��@     ��@     <�@     ��@     ��@     $�@     ̙@     �@     �@     ܕ@     ��@     (�@     ��@     `�@     ��@     �@     x�@     X�@     x�@     8�@     H�@     ��@     ��@     0�@     ��@     8�@     Ѓ@     H�@     ��@     ��@     ��@      �@     �@     ��@      @     P@     0~@      }@     �w@     �|@     �|@     �y@     �y@     �z@     �z@      x@     `y@     �u@     �u@     �u@     �u@      v@     �u@     �t@      s@     �t@     �r@      v@     �w@     Pu@     @s@     �r@     �q@     q@     0q@     pr@     0p@     `n@     �m@     0q@     �l@     Pq@     `o@      n@      k@     �l@     �g@     �i@     @k@     @l@     �h@     �k@     �j@     `h@     �f@      g@     `g@     �g@     �g@     @i@     @h@      f@     @g@     �e@     �f@      g@      e@     `c@      d@      e@      e@      b@     �f@     �a@      d@     �a@     �b@     �e@     �a@      a@      _@      `@     �`@      b@      b@     @_@     @\@     �_@     @a@      c@      j@      c@      b@     �a@     �]@     �`@      `@     �_@      ]@      \@     �]@      `@     �[@     �c@     �[@     �a@     �`@      _@     @Y@     �\@     �Y@     @Y@     �Z@     �X@     @V@     @[@     @Y@     �Y@     �\@     �Z@      \@     �\@     �W@      _@     @S@     @Y@     @S@     @T@     �V@     @T@     �R@     �K@      U@      Q@     @S@     @P@     @S@     �Y@     @V@     �b@     �V@      R@     @Q@     �S@     �Q@     @V@      S@     �P@      J@      R@      I@     �N@     �L@     �L@      M@      N@     �M@     �J@      K@      F@     �H@      L@      L@     �@@      G@      J@     �I@      G@      J@      F@     �B@      E@      B@     �@@      C@      B@     �F@     �G@      L@      O@      K@     �E@     �@@      D@      @@      A@      F@      A@      G@      C@     �A@      @@      @@      B@      B@      6@      A@      C@      <@      9@     �@@     �@     �@      3@      4@      5@      4@      6@      =@      ;@      3@      0@      1@      4@      7@      <@      7@      7@      :@      >@      ;@      8@      3@      <@      >@      ?@      8@     �A@      B@      ?@      <@     �F@      @@      >@      <@      G@     �A@      @@      A@      G@      B@     �I@      @@      G@      C@      E@     �A@      B@      I@     �F@     �D@     �E@     �C@      F@      G@     �M@      M@     �E@     �C@     �D@     �D@      K@      G@     �H@      I@      G@      G@     �K@     �L@     �N@     �Q@      K@      J@     �G@     �M@      R@      J@      Q@      P@      M@     �R@     �P@     @R@     @Q@     �T@     �P@      M@     �O@      Q@     �R@     @Q@     @T@     @Q@     @V@      T@     @R@     �P@      Y@     �S@     @T@      X@     �Y@     �S@     @V@     �T@     @W@     �S@      Z@      W@     �X@     �Y@     �]@     @[@     �\@     �\@      [@     �[@     @]@      ^@     �[@     @]@      ^@     �_@     �_@     �\@     �`@     @b@      b@      _@      a@      a@     �_@      a@      d@     �`@     `b@     �c@     �i@     �n@     @f@     `e@     �g@     �d@     �d@     `e@     �f@     `d@     �f@     @l@     �p@      m@     �f@     �f@     `h@     �e@     �f@     `g@      i@     `i@     �i@     �j@      m@     �j@      l@     �o@     Pp@      m@      o@      q@      r@     `t@     @r@     q@     Pq@     �p@     Ps@     pt@     `w@     �t@     0u@     pt@      t@     `s@      u@     �u@     Pv@     �v@     �y@      v@     �y@      @     �{@     @w@     �y@     �}@     �|@     �|@     �}@     �}@     X�@     ȁ@     �@     ��@     X�@     ��@      �@     �@     ��@     ��@     �@     ��@     P�@     �@     ��@      �@     ��@      �@     �@     Ў@     L�@     �@     �@     �@     X�@     �@     ؗ@     ̚@     ��@     �@     T�@     >�@     ��@     f�@     Z�@     ��@     ڪ@     (�@     ��@     ��@     ϳ@     ��@     %�@     Q�@     w�@     ,�@     -�@     �@     	�@     Y�@     �@      �@     ŷ@     ��@     D�@     2�@     U�@     p�@     ��@     	�@     �@     /�@     Y�@     1�@     ��@     &�@     D�@     l�@     ն@     ]�@     ��@     ��@     �@     p�@     z�@     ��@     ��@     ܥ@     �@     N�@     ��@     ��@     �@     ֥@     $�@     �@     >�@     ��@     ��@     ±@     ��@     N�@     ��@    �H�@     H�@     @�@     ��@     ��@     Ȅ@     ��@     ؀@     P{@      n@     `g@     �M@      @        
�
predictions*�	   �i�ݿ   `G�@     ί@!   I��)Z�IY8S@2���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���5�i}1���d�r�x?�x�������6�]���O�ʗ�����Zr[v��I��P=��pz�w�7����Zr[v�>O�ʗ��>����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�              �?      �?      �?       @       @      @      @      *@      .@      2@      9@      <@     �E@     �C@     �M@     �H@     �O@     �O@     @R@     �R@     �S@      S@     @S@     �R@      U@     �R@      S@     �X@     �S@     @V@     �W@     @S@     @T@     �N@      O@      P@     �O@     �H@     �J@     �E@      H@      E@      ;@      A@      ;@      6@      =@      @@      7@      7@      .@      ,@      $@      .@      $@      @      *@      @      "@       @      @      @      @      @      @      @      @       @       @      @       @      @      �?       @      �?      @       @      @      �?               @               @              �?      �?      �?               @      �?              �?              �?              �?              �?              �?      �?              �?              @              �?              �?      �?      �?              �?              �?       @       @       @      @      @      @      @      @      @      @      @      @       @      @       @      "@      @      @      3@      (@      0@      (@      1@      (@      ,@      8@      1@      9@      ;@     �@@      =@     �B@      <@      B@      F@     �B@     �B@      B@      G@      @@      H@      I@      G@     �@@      <@     �E@      B@      ;@      =@      >@      =@      @@      9@      C@      2@      8@      ?@      ;@      <@      ;@      5@      (@      $@      .@      *@      &@      $@      (@      (@      (@      (@      $@      $@       @      @      �?      @       @      @      @      @              �?       @      @              �?      �?              �?              �?        ����3      Y2Y	�	o~���A1*�g

mean squared error�{�<

	r-squared��>
�L
states*�L	   ���   `�@    �&A!T14�aI�@)^~nE�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              (@      ;@     �C@     �B@      J@     @]@     `k@     @s@     �w@     x�@     ��@     �@     V�@     d�@     ƭ@     ˲@     �@     ��@     �@     ƥ@     Z�@     6�@     ��@     V�@     ��@     �@     ��@     �@     ��@     ¥@     ��@     P�@     �@     Ԫ@     v�@     x�@     f�@     ��@     N�@     ��@     ��@     
�@     ƽ@     �@     �@     w�@     {�@     F�@     ��@     I�@     ��@     ��@     ^�@     ζ@     .�@     7�@     �@     ��@     ~�@     ��@     �@     �@     ��@     ;�@     d�@     ��@     ��@     ��@     V�@     j�@     ��@     &�@     ��@     �@     �@     V�@     �@     ��@     x�@     ��@     ��@     ��@     ؓ@     l�@     ��@     ��@     Ў@     �@     �@     ��@     ��@     `�@     0�@     ��@     �@     ȅ@     Є@     ؃@     �@     Ђ@     P�@     ��@     ��@     x�@     H�@     �}@     0~@     �}@     �z@     @{@     �y@     �y@     @x@     Py@     0y@     Pz@     �x@     x@     �v@     �w@     �t@     @u@     �v@      t@     �t@     v@     �v@     �u@     �r@     `s@     �r@     �p@     �t@     Pv@     �q@      n@     `o@     pr@     �o@     �n@      p@     �n@     �m@     @m@     �m@     `k@      l@     �j@     �m@     `l@     �j@      j@     @k@     �i@     �h@     �f@     �f@     �g@      h@     �d@     �g@     �e@     `j@      e@     �c@     �d@     @g@      c@      e@     �a@     �a@     �d@      d@      d@     @d@     �d@     �d@     �d@      b@     �`@     �_@      c@     `c@     �b@     �c@     `d@     `a@     �`@      a@     �a@     �i@     �b@     �`@      a@     �]@     �]@      `@      ]@     `b@      ^@     @Z@     @\@     �_@     �`@     �^@     �]@      Z@     �b@     �b@     �Z@     �[@     @\@     �[@      [@      X@     �V@     �X@     @Y@     @Z@     �Y@      U@     �Y@      Z@     �[@     @[@      Y@      Y@     �T@     �W@     �Q@      X@      W@      U@     �V@     �Q@      Q@     @U@     �S@     �I@     �T@      _@     �a@     �U@     �S@      P@     �R@     �R@      T@     �M@     �M@     �R@     �L@     �O@     �H@      K@     �K@     �N@      O@     �D@     �P@      K@     �G@      L@     �J@      E@     �G@     �I@      G@     �H@      H@      J@      =@     �E@     �N@      G@      D@     �G@      I@     �D@     �G@      K@      L@      M@     �D@      H@     �D@      9@     �F@      ;@     �B@     �B@      8@     �A@      @@      :@     �@@      <@      5@      =@      D@      >@      @@     ��@     Ȗ@      4@      8@      4@      8@      3@      4@      2@      A@      3@      0@      <@      1@      =@      2@      >@      A@      7@      8@      6@      7@      9@      2@      ;@      :@      <@      :@      =@      <@      8@      =@     �C@      B@      <@      A@      ;@     �A@      D@     �F@      B@      ;@      @@      <@     �B@      A@      ?@     �B@     �F@     �D@     �B@     �H@      D@      N@      H@     �C@     �J@     �K@      D@      ?@      J@     �F@      F@      J@      E@     �I@     �J@      G@      M@      J@      L@      K@     �F@     �M@      L@      L@     �Q@     @Q@     �K@     @V@      M@     �O@     �N@     �P@     �R@      R@     �O@      Q@      R@     �R@     �Q@     �J@     �R@     �T@     �R@     �Q@     @Q@     @X@     @Q@      Z@     �S@      R@     �Y@     �V@     �V@      W@     �Y@     @[@     @W@      V@     �Z@     �X@     @]@      _@      \@     @Z@     �Z@     �^@     @^@     @a@      ]@     �Y@     @_@      _@     `a@     @]@     @^@      _@     �]@     �`@     �`@     �b@     �`@     �j@     @j@     �d@     `d@     �d@     `b@     �d@     �b@     �c@     �e@     �f@     `d@     �d@      d@      g@      f@     �h@     �e@      h@     @h@     `d@      g@      h@     �n@     �i@     �j@     �l@     �l@     �k@     r@     �y@     `s@      o@      o@     `p@     �o@     �p@     �m@     �n@     �r@      r@     @s@     �s@      s@     t@     @x@     �t@      v@     �s@      u@     �s@     @t@     �s@      y@     �~@     �w@     `z@     �w@     �z@     �z@     @{@     py@     �|@      }@     �~@     P{@     ��@     0�@     Ȁ@     @�@     ��@     ��@     ��@     ��@     ��@      �@     h�@     P�@     ��@     �@     �@      �@      �@     ��@     x�@     T�@     ̒@     ��@     ��@     @�@     ��@     ̙@     ��@     �@     �@     ��@     �@     �@     �@     Ƨ@     ܪ@     ��@     l�@     [�@     ��@     �@     ϴ@     �@     [�@     `�@     �@     ˸@     �@     ��@     ��@     A�@     ��@     r�@     0�@     ��@     i�@     x�@     ��@     a�@     ��@     ��@     V�@     �@     Ҽ@     �@     ��@     ~�@     ��@     ʹ@     �@     ��@     ��@     ��@     l�@     ��@     l�@     R�@     ��@     ^�@     ��@     ¥@     ��@     ��@     ��@     d�@     ��@     ��@     r�@     ­@     �@     ��@     O�@     n�@     ��@     ��@     D�@     ��@     @�@     ؄@     �}@     �z@     `u@     pp@      G@      5@      $@        
�
predictions*�	   `�v߿   �6�@     ί@!  ��;9D@)D1s�W@2��1%���Z%�޿�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1���d�r�x?�x���FF�G �>�?�s���I��P=�>��Zr[v�>1��a˲?6�]��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?w`<f@�6v��@!��v�@زv�5f@�������:�              �?              @      �?       @      @       @      @      @      *@      1@      *@      :@     �A@      C@      A@     �J@     �N@      F@     �I@     �H@     �O@      O@     �P@     �T@     �P@      S@     �V@     �P@     �S@     �R@     �S@     �P@     �Q@      N@     �J@     �I@     �P@      D@      C@      C@      C@     �A@     �@@     �C@     �F@      ?@      6@      2@      &@      4@      &@      @      .@       @      "@      .@      @      @       @      "@      @      @      @              @      @      @       @      @      @      @       @      @      �?      @       @       @      �?      @      �?              �?       @       @              �?              �?              �?              �?              �?              �?              @       @      @              �?      �?      @       @       @       @      �?      @      @      @       @      �?      �?      @       @      @      @      @      @       @      @      &@      @      "@      *@       @      (@      &@      *@      3@      8@      (@      7@      =@      ?@     �@@      :@      7@      <@      @@     �@@     �D@      B@      E@      K@      I@     �H@     �J@     �L@     �G@     �J@     �P@     �J@     �N@      K@      L@      I@     �J@     �C@     �@@     �@@      A@     �@@     �D@      7@      5@      9@      7@      <@      2@      4@      ,@      1@      (@      .@       @      "@      &@       @      @      @      @      @      @      @      @       @      �?      @      @       @      @      @               @       @      �?              �?              �?        �(W