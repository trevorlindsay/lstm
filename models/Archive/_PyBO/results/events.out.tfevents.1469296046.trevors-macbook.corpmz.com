       �K"	  �k���Abrain.Event:2~3,���     D�0	�k���A"��
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
value	B :J*
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
:���������J
P
model/pack_1/1Const*
dtype0*
value	B :J*
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
:���������J
P
model/pack_2/1Const*
dtype0*
value	B :J*
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
:���������J
P
model/pack_3/1Const*
dtype0*
value	B :J*
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
:���������J
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
 *�
f?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"  (  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *��*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *�=*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��
�
6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�J:	�J:	�J:	�J
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
:	�J
�
2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�J
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel/zeros2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�J
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�J
�
/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�J
�
1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�J
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�J
s
.model/RNN/MultiRNNCell/Cell0/dropout/keep_probConst*
dtype0*
valueB
 *�
f?*
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
:	�J
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell0/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/sub*
T0*
_output_shapes
:	�J
�
3model/RNN/MultiRNNCell/Cell0/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
:	�J
�
(model/RNN/MultiRNNCell/Cell0/dropout/addAdd.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob3model/RNN/MultiRNNCell/Cell0/dropout/random_uniform*
T0*
_output_shapes
:	�J
�
*model/RNN/MultiRNNCell/Cell0/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell0/dropout/add*
T0*
_output_shapes
:	�J
�
(model/RNN/MultiRNNCell/Cell0/dropout/InvInv.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell0/dropout/mulMul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0*
_output_shapes
:	�J
�
*model/RNN/MultiRNNCell/Cell0/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mul*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0*
_output_shapes
:	�J
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB"�   (  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *��*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *�=*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�J:	�J:	�J:	�J
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
:	�J
�
2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�J
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel/zeros_22model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�J
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�J
�
/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�J
�
1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�J
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�J
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�J
s
.model/RNN/MultiRNNCell/Cell1/dropout/keep_probConst*
dtype0*
valueB
 *�
f?*
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
:	�J
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell1/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/sub*
T0*
_output_shapes
:	�J
�
3model/RNN/MultiRNNCell/Cell1/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
:	�J
�
(model/RNN/MultiRNNCell/Cell1/dropout/addAdd.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob3model/RNN/MultiRNNCell/Cell1/dropout/random_uniform*
T0*
_output_shapes
:	�J
�
*model/RNN/MultiRNNCell/Cell1/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell1/dropout/add*
T0*
_output_shapes
:	�J
�
(model/RNN/MultiRNNCell/Cell1/dropout/InvInv.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell1/dropout/mulMul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0*
_output_shapes
:	�J
�
*model/RNN/MultiRNNCell/Cell1/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0*
_output_shapes
:	�J
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
:	�J
d
model/Reshape/shapeConst*
dtype0*
valueB"����J   *
_output_shapes
:
e
model/ReshapeReshapemodel/concatmodel/Reshape/shape*
T0*
_output_shapes
:	�J

model/dense_wVariable*
dtype0*
shape
:J*
	container *
shared_name *
_output_shapes

:J
�
.model/dense_w/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@model/dense_w*
valueB"J      *
_output_shapes
:
�
,model/dense_w/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *��*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *�=*
_output_shapes
: 
�
6model/dense_w/Initializer/random_uniform/RandomUniformRandomUniform.model/dense_w/Initializer/random_uniform/shape*
_output_shapes

:J*
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

:J
�
(model/dense_w/Initializer/random_uniformAdd,model/dense_w/Initializer/random_uniform/mul,model/dense_w/Initializer/random_uniform/min* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:J
�
model/dense_w/AssignAssignmodel/dense_w(model/dense_w/Initializer/random_uniform*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes

:J
x
model/dense_w/readIdentitymodel/dense_w* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:J
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
 *��*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *�=*
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
valueB�J*    *'
_output_shapes
:�J
�
model/VariableVariable*
dtype0*
shape:�J*
	container *
shared_name *'
_output_shapes
:�J
�
model/Variable/AssignAssignmodel/Variablemodel/zeros_4*
validate_shape(*!
_class
loc:@model/Variable*
use_locking(*
T0*'
_output_shapes
:�J
�
model/Variable/readIdentitymodel/Variable*!
_class
loc:@model/Variable*
T0*'
_output_shapes
:�J
�
model/Assign/value/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�J*
T0*
N
�
model/Assign/value/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�J*
T0*
N
�
model/Assign/valuePackmodel/Assign/value/0model/Assign/value/1*'
_output_shapes
:�J*
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
:�J
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
:	�J
�
*model/gradients/model/MatMul_grad/MatMul_1MatMulmodel/Reshape&model/gradients/model/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:J
d
(model/gradients/model/Reshape_grad/ShapeShapemodel/concat*
T0*
_output_shapes
:
�
*model/gradients/model/Reshape_grad/ReshapeReshape(model/gradients/model/MatMul_grad/MatMul(model/gradients/model/Reshape_grad/Shape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/gradients/model/Reshape_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�J
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
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
:���������J
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1Mulmodel/zeros_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�J
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�J
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
:	�J
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�J
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
:	�J
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
��*
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
��
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
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
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
:	�J
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice_1SliceTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul\model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffset:1Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:���������J
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
:	�J
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
:	�J
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�J
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
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
:���������J
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1Mulmodel/zerosMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�J
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
:	�J
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
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�J
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
:	�J
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�J
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�J
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�J
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
:	�J
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�J
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
:	�J
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
��*
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
��
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
:�
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
��
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
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_2MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_1Identitymodel/clip_by_global_norm/mul_2*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_3MulVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_2Identitymodel/clip_by_global_norm/mul_3*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_4MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_3Identitymodel/clip_by_global_norm/mul_4*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_5Mul*model/gradients/model/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:J
�
6model/clip_by_global_norm/model/clip_by_global_norm/_4Identitymodel/clip_by_global_norm/mul_5*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:J
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
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_6Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
\
model/zeros_7Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
\
model/zeros_8Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
f
model/zeros_9Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
_output_shapes
:
��
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_10Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
_output_shapes
:
��
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_11Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_12Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
c
model/zeros_13Const*
dtype0*
valueBJ*    *
_output_shapes

:J
�
model/model/dense_w/AdamVariable*
	container *
_output_shapes

:J*
dtype0*
shape
:J* 
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

:J
�
model/model/dense_w/Adam/readIdentitymodel/model/dense_w/Adam* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:J
c
model/zeros_14Const*
dtype0*
valueBJ*    *
_output_shapes

:J
�
model/model/dense_w/Adam_1Variable*
	container *
_output_shapes

:J*
dtype0*
shape
:J* 
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

:J
�
model/model/dense_w/Adam_1/readIdentitymodel/model/dense_w/Adam_1* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:J
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
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_2*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_3*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
)model/Adam/update_model/dense_w/ApplyAdam	ApplyAdammodel/dense_wmodel/model/dense_w/Adammodel/model/dense_w/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_4* 
_class
loc:@model/dense_w*
use_locking( *
T0*
_output_shapes

:J
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
:�J*
T0*
N
�
model/HistogramSummary/values/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�J*
T0*
N
�
model/HistogramSummary/valuesPackmodel/HistogramSummary/values/0model/HistogramSummary/values/1*'
_output_shapes
:�J*
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
value	B :J*
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
:���������J
R
model_1/pack_1/1Const*
dtype0*
value	B :J*
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
:���������J
R
model_1/pack_2/1Const*
dtype0*
value	B :J*
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
:���������J
R
model_1/pack_3/1Const*
dtype0*
value	B :J*
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
:���������J
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
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�J:	�J:	�J:	�J
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
:	�J
�
4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�J
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel_1/zeros4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�J
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�J
�
1model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�J
�
3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�J
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�J
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
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�J:	�J:	�J:	�J
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
:	�J
�
4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�J
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel_1/zeros_24model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�J
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�J
�
1model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�J
�
3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�J
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�J
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�J
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
:	�J
f
model_1/Reshape/shapeConst*
dtype0*
valueB"����J   *
_output_shapes
:
k
model_1/ReshapeReshapemodel_1/concatmodel_1/Reshape/shape*
T0*
_output_shapes
:	�J
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
valueB�J*    *'
_output_shapes
:�J
�
model_1/VariableVariable*
dtype0*
shape:�J*
	container *
shared_name *'
_output_shapes
:�J
�
model_1/Variable/AssignAssignmodel_1/Variablemodel_1/zeros_4*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking(*
T0*'
_output_shapes
:�J
�
model_1/Variable/readIdentitymodel_1/Variable*#
_class
loc:@model_1/Variable*
T0*'
_output_shapes
:�J
�
model_1/Assign/value/0Pack2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�J*
T0*
N
�
model_1/Assign/value/1Pack2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�J*
T0*
N
�
model_1/Assign/valuePackmodel_1/Assign/value/0model_1/Assign/value/1*'
_output_shapes
:�J*
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
:�J"	M���      kxQ�	��k���A*�9

mean squared errorW�C=

	r-squared �B:
�+
states*�+	    ���   �L�@    c2A!�i\�08��)�8Zi A2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr����8"uH>6��>?�J>4�j�6Z>��u}��\>BvŐ�r>�H5�8�t>u��6
�>T�L<�>
�}���>X$�z�>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              $@     �r@     ��@      �@     v�@     ��@     ԟ@     p�@     d�@      �@     ��@     N�@     ��@     ��@     ��@     f�@     ��@     (�@     ��@    ���@    �4�@    ���@     ��@    �/�@     �@     ��@     c�@     ��@    �'�@     ��@     ��@     e�@     �@     E�@     ܽ@     ��@     �@    ���@    ���@    ���@    ���@    ���@    ��@    @�@     .�@    ���@    ���@     N�@    � �@     �@    ���@    �:�@     ��@    �D�@    ���@     ;�@     ��@    ���@     Կ@     E�@     H�@     V�@     ��@     �@     0�@     ��@     ��@     ��@     >�@     �@     2�@     d�@     Р@     d�@     T�@     �@     p�@     D�@     �@     d�@     ��@     �@     ��@      �@     �@     x�@     ��@      �@     ��@      ~@     }@     �x@     �x@     �u@     �s@      s@     `q@      m@     �k@     �k@     @e@     �e@      b@      e@     �`@     �]@     �]@     @Y@      U@     @T@     �P@     �Q@      P@      I@     �F@      K@      C@     �B@      ;@     �A@      <@      3@      5@      6@      5@      2@      3@      .@      ,@      ,@      &@       @       @      &@       @       @      @      "@      @      @      $@       @      @      @       @      @       @      @       @              @              �?              �?      @      �?      �?              �?      �?              �?       @              �?              �?              �?              �?              �?              �?               @              �?               @      �?      �?      �?       @      �?      @      �?      @       @      @      @      @      @       @       @      @      @      @      "@      &@      $@      @      @      @      "@      *@      "@       @      1@      .@      1@      2@      2@      3@      6@      8@      >@      =@      E@     �B@     �D@     �E@      P@     �M@      P@      V@     @U@     �S@     @]@     �X@     @X@     �a@      ]@     �b@      d@     �f@     �g@     `j@     �m@     p@     �r@     �t@     ps@     �u@     �y@     �|@     �~@     ؀@     ��@     ��@     ��@     ��@     ��@     �@     �@     0�@     �@      �@     \�@     ��@     H�@     �@     <�@     ��@     6�@     ��@     P�@     �@     p�@     Ͱ@     ��@     ��@     ��@     _�@     '�@     �@    �S�@     �@    �\�@     �@    ���@     ��@     	�@    ��@     3�@    �2�@     ��@     ��@    �d�@    @�@     ��@    �V�@    ��@    �T�@    �\�@    ���@    ��@    �X�@     ��@    ���@    ���@     ��@    �]�@    ��@    ��@     �@     ��@    � �@    �&�@    ���@     ��@     �@     )�@     ��@     P�@     ��@     ��@     Ľ@     Ͻ@     G�@     ݻ@     Ϲ@     F�@     ��@     ��@     d�@     n�@     $�@     ��@     Pq@     �q@     pw@      ~@     І@     P�@     �x@      &@        
�
predictions*�	   �sؑ�   @��x?     ί@!  �!@�)�O��T�?2��Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��>h�'��O�ʗ�����Zr[v��})�l a�>pz�w�7�>>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?�������:�              �?      5@     �Q@      b@     Pp@     �u@     �|@     `~@      {@     @y@     �r@     �k@     �d@     �`@     �V@     �J@     �O@     �G@      @@      A@      5@      6@      ,@      8@      ,@       @      $@      ,@      $@      "@       @      @       @      @      @      �?      @      @      @      @               @       @      @      �?              �?      �?      �?              �?              �?       @              �?              �?              �?      �?               @               @      �?              �?              �?      �?      �?              �?       @      �?      @      �?       @       @       @      @      @      @               @      @      @       @      @      @      @      @      @       @      @       @      @      @      @      �?              �?       @        �~��       ����	�
l���A*�A

mean squared error��C=

	r-squared �>;
�6
states*�6	    �p�   �z�@    c2A!���2���)D�1�v�@2��Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�����W_>�p
T~�;��z��6��so쩾4���o�kJ%�4�e|�Z#�RT��+��y�+pm�;3���н��.4Nν�8�4L��=�EDPq�=���6�=G�L��=nx6�X� >�`��>RT��+�>���">��f��p>�i
�k>4�e|�Z#>��o�kJ%>�'v�V,>7'_��+/>_"s�$1>�so쩾4>�z��6>u 5�9>p
T~�;>��8"uH>6��>?�J>������M>28���FP>Fixі�W>4�j�6Z>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�               @       @      ,@      _@     ��@     �@      �@     ĝ@     ��@     �@     ��@     �@     v�@     \�@     Ы@     ��@     �@     }�@     �@     x�@     ��@     ^�@     d�@     ��@     >�@     ڻ@     ؽ@     �@     p�@     ��@     ��@     ��@     5�@     S�@     �@     +�@     �@     �@    �~�@     `�@    ��@     7�@     ��@    ���@     �@    �0�@    �{�@    ���@    ���@    ���@    @��@    ���@     ��@    ���@     ��@    �w�@     ��@     p�@     ��@     ��@    ���@     �@     ��@    �.�@    ���@     ��@     N�@     _�@     ��@     ^�@     G�@     �@     �@     h�@     ��@     l�@     ��@     v�@     b�@     H�@     *�@     X�@     ��@     ��@     �@     ��@     ��@     H�@     |�@     ��@     ��@     Ȏ@     (�@     @�@      �@     P�@     �@     ��@     ��@     �@     0@     P|@     �x@      z@     �u@     �w@     �r@     @p@     �p@     `n@     �j@      k@     �e@     �d@      b@     �b@      c@      c@      `@     �W@     �V@     �V@     �R@     �S@     @T@      Q@     �O@     �N@     �F@     �C@     �J@     �G@      B@     �@@      >@      ?@     �D@      :@      5@      2@      5@      1@      1@      &@      1@      "@      .@      &@      0@      &@       @      $@      "@      @      @      (@      $@      @      @      @      @      "@       @       @      @      �?      @      @      @      @       @      @      @      @      �?      �?               @      �?      �?       @       @      �?      �?       @      �?              �?              �?      �?              @      �?              �?              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @               @              �?      �?               @       @      �?              �?              �?              �?              �?              �?       @       @      �?      �?      �?       @               @              @      @      �?       @      �?      �?      �?      @       @      �?      @      @       @      @      @      @      @      @      @      @      @      @      @      @      &@      "@      &@      @      1@      @      "@       @      2@      (@      3@      5@      0@      1@      2@      8@      6@      ?@      9@      ?@      ;@      ?@      A@     �C@     �E@     �J@     �O@      N@      N@      M@     @Q@     �U@     �V@     @U@     @X@     @\@     @\@     @Z@      b@      a@     �g@     �f@     �h@     �g@     �j@     �n@     @n@     �q@     �o@     @r@     �s@      v@     `y@     �x@     p|@     �~@     P�@     p�@     @�@     ��@     ؅@     H�@      �@     Ȋ@     0�@     ��@     ��@     h�@     ��@     L�@     ��@     `�@     ԙ@     �@     �@     @�@     :�@     �@     ��@     �@     F�@     H�@     ª@     J�@     Z�@     �@     ɲ@     δ@     ��@     {�@     B�@     z�@      �@    ��@     �@     [�@     ��@     ��@    ���@    �E�@    �B�@     i�@     ��@     M�@    ���@    ���@    ���@     ��@    ���@     C�@    ���@     8�@     ��@     ��@    ���@    ��@     P�@    �a�@     .�@     n�@     ��@     L�@     ٵ@     ?�@     ��@     +�@     ��@     =�@     ²@     �@     ��@     ��@     ��@     ɳ@     �@     1�@     ��@     Ա@     ��@     F�@     7�@     g�@     ǳ@     �@     Ы@     X�@     h�@     @�@     h�@     8�@     (�@     ��@     h�@     �}@     @n@      K@      @        
�

predictions*�
	    ��h�   @`c�?     ί@! �f���P@)0
@"[��?2�P}���h�Tw��Nof�5Ucv0ed�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���VlQ.��7Kaa+��vV�R9��T7����S�F !?�[^:��"?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?�������:�               @      �?              �?               @      �?      �?              �?              �?              @              �?       @              �?              �?              �?              �?              �?      �?              �?      �?      �?       @      @              @      �?      @      @      @      @       @      @      (@      @      1@      5@      :@      ?@     �A@     �I@      H@      S@     �S@     �[@     �_@      b@     �b@      d@     �h@      n@     �p@     �t@     @t@      t@     t@      n@     @j@     �e@     �\@     �U@     �E@      >@       @      @        m�'b.      X
�	j�+l���A*�\

mean squared error )@=

	r-squared@��<
�G
states*�G	   `��   @*�@    c2A!x���jj��)�!h���@2�#�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�Ľ�
6������Bb�!澽5%�����EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����-���q�        �-���q=���_���=!���)_�=����z5�=���:�=��s�=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=�8�4L��=�EDPq�=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�#              @      2@     �N@      o@     `}@     ��@     ��@     �@     x�@     ��@     `�@     ��@     &�@     �@     �@     �@     U�@     �@     ڹ@     ��@     Z�@     ��@     U�@     X�@     ��@     Ķ@     f�@     ��@     +�@     �@     ��@     ��@     ˾@    �h�@    �8�@    ���@    �#�@      �@    ���@    �X�@    ���@     ��@     ��@     9�@     �@     �@    ���@    @�@     ��@    �y�@    �F�@     ��@    ���@     ��@     ��@    �0�@     ��@    ���@    ��@     |�@     .�@     ��@     /�@     A�@     6�@     ��@     Y�@     |�@     �@     �@     j�@     r�@     ֪@     ��@     H�@     ��@     ��@     ��@     R�@     \�@     ̚@     ��@     ؗ@     ��@     ��@     <�@     $�@     ؒ@     ��@     x�@     h�@     ��@     ��@     ��@     ��@     ��@     ��@     (�@     Ё@     �@     �@     �}@     �z@     �z@      x@     �y@      w@     �v@     `s@     0t@     �r@      p@     0p@     @k@      j@     �k@     �j@     �h@     �g@     `g@     �d@      d@      [@     @\@     �^@     �\@     �X@     �Z@     �X@      X@     @V@     �W@     @S@     @R@     @Q@     �O@     �M@      K@      I@     �K@      E@     �D@     �G@      D@     �H@      D@      C@      ;@      @@      B@      ;@      9@      ;@      7@      9@      <@      7@      0@      "@      6@      .@      *@      @      "@      ,@      $@      *@      5@      @      .@      "@      @      (@      @       @      @      1@      @      @      "@      @      @      "@      @       @      @      @      �?      @      @      @      @      @      @      $@       @      @       @      �?       @       @       @      @      �?      @      �?      �?      @       @      @      @      @       @      @      @       @              @      @      @      �?       @      �?      @       @      �?              �?       @      @      �?      @       @      �?      �?      @              �?      �?              �?              �?      �?       @              @              �?       @              �?      @      �?      �?       @      �?      @              @              �?      �?       @               @      �?              �?              @      �?      �?      �?      �?               @               @              �?      �?              �?              �?      �?              &@      0@               @      �?              �?              �?       @      �?      �?               @              �?              �?              �?              �?      �?       @              �?       @      �?              @      �?      �?      �?       @       @      �?       @      �?      @      @               @      �?       @               @       @      �?      �?              �?      �?              @      @      @              @      �?       @      @      @              �?      �?      �?       @      @      @      �?       @      @       @       @       @      �?      �?      @       @      @       @      @      @      @      @      @      �?      @      @      @      @      @      @      @       @      @      @      @      @      @      @      @      @      @      @      @      &@      @      *@      @      "@      "@      @      @      .@      .@      *@      4@      *@      2@      .@      5@      0@      1@      5@      4@      6@      5@      9@      3@     �C@      B@      7@      B@      :@     �D@      D@     �E@     �J@     �N@      J@     @Q@     �S@     @Q@     �V@      T@      R@      V@     @U@     �W@     @Z@     �X@     �Z@     �Z@     �_@     �a@     �b@     `b@      h@     @g@      e@     @h@     @g@     `k@     �l@     `k@     pp@     @p@     `p@     `r@     �r@     u@     `w@     0t@     pw@      {@     �|@     0{@     }@     @~@     ��@     ؃@     `�@     x�@     X�@     ��@     H�@     P�@     ȉ@     ȍ@     p�@     ��@     ��@     (�@     В@     Ȕ@     d�@     �@     ��@     ��@     d�@     ��@     |�@     ��@     `�@     ��@     ��@     |�@     �@     ��@     ��@     [�@     �@     �@     ��@     �@     ��@      �@     ��@     r�@     ¿@    ���@     T�@     ��@     D�@     ��@    ���@    ���@     ��@     	�@     O�@     �@    �]�@     
�@     Z�@    ���@     ��@    �%�@     ��@    ���@    �/�@    �c�@     "�@    ��@    �q�@    ��@     ��@    ��@     ��@     B�@     �@     +�@     M�@     ��@     ��@     b�@     ߴ@     ô@     ��@     �@     B�@     0�@     l�@     I�@     �@     �@     2�@     ,�@     [�@     .�@     �@     0�@     �@     �@     ��@      �@      v@     �i@      F@      "@        
�
predictions*�	   �Ri��    �;�?     ί@!  0��%	�)^���\@2�
��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s����h���`�8K�ߝ뾢f����>��(���>��Zr[v�>O�ʗ��>>�?�s��>����?f�ʜ�7
?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�
              @      *@     �@@      H@      U@      ]@     �Z@     �^@     �\@      [@     �Z@     �T@     �V@     @R@     �U@     �N@     �N@     �J@     �K@      H@     �E@      D@     �H@     �B@      D@     �@@     �A@      8@      3@      6@      8@      8@      2@      .@      1@      @      0@      (@      "@      "@       @      $@      @      @      @      @      @       @      @      @      @      @      �?      @      @      @               @      �?      �?      �?       @              �?      �?              �?               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?       @              �?               @      �?      �?              @              @      @      �?      �?       @       @       @      @      @      @       @      @      @      @      @      @      "@      &@      &@      @      .@      "@      5@      9@      4@      8@      8@      >@      @@      B@      >@      B@      =@      ?@      I@      K@     �N@      Q@      P@      M@     @S@     �V@     @Q@      Y@     @U@      V@     @V@     @Q@      T@     �V@      U@     �M@      N@      J@     �C@      :@      3@      "@               @        �he��/      @�X�	$�Jl���A*�_

mean squared error�v?=

	r-squared���<
�J
states*�J	    �M�   �@    c2A!��֎Է@)r��
8�@2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ���>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              �?      &@     �@@      f@     pz@     ��@     ��@     P�@     X�@     �@     ��@     d�@     �@     �@     ��@     ��@     \�@     �@     ��@     7�@     _�@     �@     ��@     �@     �@     з@     �@     �@     ֻ@     n�@    ��@    ���@    �]�@     ��@    ���@    ���@    ���@    ���@     Q�@    ���@    �~�@    ���@    �z�@    ���@    �d�@     ��@    ���@     H�@    �8�@     �@     O�@     ,�@     0�@     ��@     ��@     0�@    ��@     ��@     =�@     ��@     �@     �@     '�@     a�@     ��@     y�@     |�@     �@     r�@     ��@     ک@     x�@     4�@     b�@     6�@     ڡ@     L�@     ��@     ��@     t�@     0�@     �@     8�@     `�@     ��@     0�@     �@     D�@     ��@     ��@     ��@     ؋@     Њ@     ��@     �@     p�@      �@     �@     ��@     ȁ@     ��@     ��@     �}@     �@     �|@     �|@      z@     Pw@     �u@     �u@     �t@     �t@      t@     �s@     Pr@     �p@     Pp@     Pp@     �l@     �k@     `h@     �j@      j@     @h@     �g@     @e@     �d@     �c@      e@      a@     @`@     @\@     �_@     �`@     @]@      Y@      W@     @T@     @U@     �V@     @S@     �S@     �N@     �N@      S@      L@      P@     �G@     �L@     �J@     �F@      A@     �F@     �D@     �C@      B@     �B@      =@      D@      <@      A@      C@      3@      <@      2@      ;@      0@      =@      0@      4@      4@      3@      0@      9@      &@      1@      &@      4@      1@      &@      3@      .@      *@      "@      (@      &@      @      (@      (@      $@      *@      $@      @      1@      "@      @      @       @      @      $@      �?      @      @      @      @      @      @       @      @       @      @       @      @      "@      @      @      @      @      @      @       @      �?      �?      @      @       @      @       @      @      @      @      �?      @       @      �?               @      @       @      �?       @      @      @              @       @       @      @      �?      �?       @              @      �?      @       @      �?       @       @      @       @      �?      �?       @      �?      @      @      �?              �?      �?      �?       @       @      �?      �?              @       @              @      �?       @       @      �?       @      �?      �?      �?      �?      �?      �?              �?               @       @      �?      D@      F@               @              @      �?       @       @              �?       @               @              �?      �?      �?       @               @               @       @      �?       @       @       @      �?       @       @              �?               @      �?      @               @      @      @      �?      �?      @               @      �?       @       @       @      @       @      @      �?      �?       @      @      @       @      �?      �?       @       @      @      @      @      @      �?       @      @      @      @      @       @      @      @      @       @      @      @      @       @      @      @      @      @      @      @      @      @       @      @      @      @      "@      "@      @      @      @      &@      $@       @      $@      (@      "@      "@      .@      @      (@      @      $@       @      @      @      $@      1@      0@      $@      7@      (@      ,@      ,@      .@      ,@      *@      $@      9@      ,@      7@      7@      6@      5@     �@@      <@      A@      B@      F@      @@      D@      E@     �E@      K@      N@      F@      D@     �N@     �O@     @Q@      O@      O@     �N@      R@      X@      X@     �Z@     �W@      \@     �^@     �[@     ``@     �]@     �]@     @`@     @`@      c@     �c@      d@     �b@     �f@     �f@     `j@     �m@     �h@     @h@     `l@     Pp@     `o@      q@     @q@     �r@     �q@     `s@     �u@      w@     y@      {@     ��@     �}@     �}@     �~@     �|@     ��@     ��@     `�@     ��@     �@      �@     �@     H�@     h�@     ��@     ��@     �@     h�@     ��@     �@     ��@     �@     ��@     ̖@     0�@     Ě@     Ԝ@     h�@     ��@     @�@     ��@     ģ@     ��@     ��@     h�@     ��@     ��@     ��@     }�@     V�@     �@     �@     �@     ��@     ϸ@     ǹ@     ��@     2�@     ��@     f�@     ^�@    ���@     ��@    ���@     ��@     �@    ���@     l�@     H�@    � �@    ���@     ��@    ���@    �K�@     2�@     j�@     ��@     ��@    �I�@    �N�@     ��@    ��@    �2�@     p�@    �M�@    �0�@     ��@     ��@     8�@     �@     �@     u�@     �@     �@     9�@     k�@     r�@     �@     �@     ��@     �@     ,�@     ��@     t�@     ܸ@     ��@     :�@     `�@     `�@     �@     `�@     `|@     �s@      e@     �B@      @        
�
predictions*�	   ����    ��?     ί@!  ��S�0�)�;#� @2�	!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���.����ڋ���[���FF�G �>�?�s���8K�ߝ�>�h���`�>��[�?1��a˲?6�]��?����?�vV�R9?��ڋ?�.�?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�	              @      @      :@      A@     �T@     @]@     �`@      b@     �_@     @_@     @]@     �Z@     @[@     �Q@     �N@     �M@     �O@     �K@      K@      H@      G@     �B@     �C@      :@      >@      6@      =@      ;@      8@      1@      9@      0@      .@      &@      0@      *@      (@      @      &@       @      @       @      $@      &@      @      @      @      @      @       @      @      @      @      @      @      @       @       @       @              �?      �?              �?       @              �?      �?              �?      �?      �?              �?      �?              �?              �?               @              �?       @              �?               @              �?      @      �?       @      �?       @       @      �?      �?       @       @       @      @      @      @      @      @      &@      @       @      @      "@      @      &@       @      0@      ,@      "@      1@      *@      4@      ;@      (@      <@      C@      5@      =@     �@@      B@      F@      K@      H@      Q@      O@     @P@     �Q@     @S@      S@      T@     �U@     @V@      Z@     �Y@     �V@     �U@      S@     �P@     �F@     �D@     �B@      <@      .@      0@      2@      $@      @              �?        �ʊ��0      z�{	��cl���A*�a

mean squared errorA�==

	r-squared��=
�K
states*�K	    b��   �y@    c2A!�.���¹@)K ����@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=:[D[Iu='1˅Jjw=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=����z5�=���:�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%               @      "@      L@     �_@     �h@     ��@     �@     ��@     D�@     h�@     ʠ@     &�@     j�@     ��@     k�@     ��@     `�@     ��@     J�@     ��@     $�@     D�@     ��@     ��@     ̵@     	�@     �@     {�@     n�@     ��@     >�@     ֿ@    ���@    �Q�@    ���@     J�@     �@    �w�@     x�@    �R�@     2�@    �`�@    �d�@    ���@     ��@    ���@    ���@    ���@     g�@    ��@     ,�@    ���@     "�@    �%�@    ���@     {�@    ���@     ��@    ���@     :�@     ��@     ��@     6�@     +�@     `�@     ʲ@     ��@     �@     V�@     8�@     �@     ֩@     h�@     ��@     B�@     \�@     v�@     ��@     ��@     ؜@     �@      �@     ��@     ��@     �@     �@     8�@     ؑ@     ��@     `�@     ��@     �@     ��@     ��@     �@     p�@     0�@     X�@     ��@     0�@     ��@     8�@     ��@     �@     �@     @     �{@     �x@     �z@     �x@     �x@     �v@      w@     �r@     Ps@     u@     �q@     �q@     �r@     �p@      p@     �o@     �m@     �k@     @h@     �n@     �e@     `g@     �e@     �c@     �f@      c@     �`@      c@     �b@     �a@     �a@     �]@      Y@     @U@     @Z@     �V@     �X@     @T@      X@      T@     �R@      T@      Q@     �R@     �Q@      Q@     �J@     �P@     @P@     @P@      G@      N@     �D@      D@     �D@     �H@      D@      D@      @@     �D@     �A@     �C@      7@      ?@      A@      ;@      B@      9@      <@      ;@      6@      :@      0@      5@      6@      *@      2@      4@      .@      6@      "@       @      *@      .@       @      @      (@      *@      *@      @      $@      @      $@      5@      @      @      .@      @      "@      &@      (@      @      @       @      "@      @       @      @       @      @       @      @      @      @      @       @      &@      @      @      @      @      @      @      @      @      @      @       @      @      @      @      �?      @      �?      @      @              �?      @      @      @              @      �?      @      @      @       @      @      @      �?      @      @              �?      @              @      �?               @      @              @      @      �?              @               @      @      �?      �?      @       @               @       @      �?               @       @       @      �?      �?      �?      �?              �?      �?              @      @       @      �?      N@      G@              �?               @              �?      �?      �?      �?              �?              �?              �?      �?              �?      �?      @      @       @      @      �?      @      �?      @      �?              �?      �?              @      �?      �?               @      @       @      @      �?      @       @       @       @       @       @      �?      �?      @      �?      @      @              @      @       @      @      @      @      @      @       @       @      @      �?       @      @      @      @      @      @      @      "@      @      @      @      @       @      @      @      @      @      @      @      (@       @      @      @       @      @      @      @      &@      @      @       @       @      @      &@      (@      &@      *@      1@      $@      "@      *@      (@      *@      3@      $@      *@      .@      (@      $@      (@      5@      ,@      1@      0@      6@      9@      6@      =@     �B@      :@      7@      A@     �B@      B@     �E@      H@     �C@     �G@      G@      F@     �M@     �G@     �I@      L@      L@     @P@     �P@      L@     �M@     �P@     @V@     �Q@     �S@      X@     �X@     �[@     �Y@     �X@      \@     �Z@     �Z@     ``@      a@      b@      `@      d@      d@     �c@     `d@     �d@     �f@     �i@     �g@     �j@     `l@     `m@     �m@      q@     �q@     �q@     Pr@     �t@     �z@     v@     0v@     0w@     �v@     �z@     �{@     �{@     �}@     @}@     �@     (�@     ��@     (�@     ��@      �@     �@     ��@     Ȋ@     ��@     ��@     ��@     ��@     ��@     <�@     ��@     ��@     ��@     D�@     ��@     �@     ,�@     ��@     �@     ��@     <�@     ,�@     ��@     .�@     @�@     Ħ@     2�@     �@     L�@     ��@     ��@     ��@     ��@     ��@     7�@     ��@     H�@     ��@     ��@     ]�@     �@     l�@    �'�@      �@    ���@     n�@    ���@    ���@    ���@     ��@     ��@    �[�@    �b�@     L�@     ��@     E�@     l�@     b�@    �F�@     ��@    �;�@    �"�@    �V�@    �8�@     M�@     ��@     ��@    �9�@     ��@     ��@     �@     �@     5�@     ��@     ~�@     O�@     !�@     ��@     Թ@     ��@     ��@     �@     ��@     =�@     �@     �@     �@     ��@     �@     ֥@     ��@     0�@      �@     ��@     Px@     @e@      F@       @      @      @        
�
predictions*�	   `���   ����?     ί@!  �pK@)����HV@2�
���g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'�������6�]�����[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=���_�T�l׾��>M|Kվ�ߊ4F��>})�l a�>>�?�s��>�FF�G ?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�
              @      @      >@     �S@      ^@     �c@     `e@     @d@     `b@     �`@      `@     �]@     �[@     @Z@      V@     �R@      P@      Q@      G@      A@      F@      B@      C@      8@      =@      6@      6@      3@      4@      4@      (@      *@      0@      &@      $@      ,@      @      @      $@      @      @      @      @      @      @      @       @      �?              @      �?      �?      @      �?      @      �?      @      �?       @              �?      �?              �?              �?      �?      �?      �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @       @       @      �?               @      �?      �?       @              @      @       @      @      @      @      @      @      "@      @      @      &@      @      �?       @      &@       @      &@      0@      .@      1@      "@      3@      9@      7@      <@      ?@      B@      >@     �F@      L@      T@      P@     �Q@     @R@      K@     �M@     �L@     �H@      F@      L@     �M@     �F@     �P@      M@     �G@      N@     �D@      D@      B@     �D@     �B@      A@      =@      5@      1@      ,@      *@      3@      &@       @      @       @        -��B0      y5^�	�}l���A*�`

mean squared error3�:=

	r-squared��@=
�K
states*�K	   ��%�   ��R@    c2A!h2�gR�@)�}�Ӵ��@2�%w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=��
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%               @      4@     �Q@     `t@     �@     h�@     X�@     $�@     Ȟ@     Ĳ@     ³@     ��@     ��@     ��@     *�@     ��@     7�@     A�@     �@     ö@     ;�@     ɷ@     ȸ@     Z�@     ͷ@     ��@     K�@     N�@     �@    �%�@    ���@    �l�@     �@    �J�@    ���@    ���@    ���@     7�@    ���@    ���@     }�@     1�@     �@    ��@    ��@    �s�@    ���@    ���@     ��@     	�@    ��@    �8�@     ��@     =�@     $�@     ��@     P�@     �@     ��@     \�@     ��@     �@     �@     ��@     �@     j�@     J�@     ګ@     ��@     �@     "�@     �@     ��@      �@     �@     .�@     |�@     x�@     ܜ@     ��@      �@     x�@     ,�@     ��@     �@     Ȓ@     �@     ��@     ��@     ��@     �@     8�@     8�@     X�@     8�@     І@     ��@     h�@      �@      �@     0�@     ��@     ��@     ��@     �@     �|@     0{@     �y@     0z@     �x@     �x@     @x@     �u@     0t@     ps@     �s@     0s@     @r@     `o@      o@     �r@     `l@     �p@     �k@     @n@      l@     �g@     �k@     @f@      f@     `f@      f@     �d@      a@      b@      b@     �e@     @_@     �`@      `@     ``@     @_@      `@     @Z@     @_@     @]@      S@      U@      Y@     @V@     �U@      T@     @Q@     �K@     �Q@     �Q@     �Q@      Q@     @T@      H@      J@      J@     �I@      H@      I@      D@      F@      E@      F@     �@@      @@     �B@      B@     �E@     �B@      ;@     �A@      ?@      8@      3@      0@      7@      7@      5@      8@      7@      5@      .@      3@      (@      &@      5@      *@      3@      3@      $@      (@      0@       @      *@      &@      @      ,@      "@       @      &@      $@      "@      &@      "@      $@       @      @       @       @      &@      "@      "@      @      @      @      &@      @      @      "@      @      @      @       @      @      @      &@       @      @      @      @      @      @      @      @      @       @      @      @      @      @       @      @       @      @       @      @       @      @      @              @               @       @       @      @      @      @      @      �?      �?               @      @      �?               @              @              �?      �?      @       @       @       @      �?       @      �?       @       @      �?              @       @       @      @               @      �?       @             �T@      M@              �?      �?               @              �?              �?       @               @               @               @               @               @       @       @      @       @      �?      @      @      @      @       @      �?       @      �?       @               @      @      @      �?      @      @               @      @      @      @      @      @      �?      @      @      @              �?      @      @       @       @      @      @      @      @      @      @      @      @      @      @      @      @      $@      @      @      "@      @      @       @      @      @      @       @      @      @      @      @      @      @      @      @      (@       @       @      "@       @      "@      @      &@      &@      $@      &@      1@      *@      &@      &@      &@      $@      $@      ,@      2@      6@      1@      .@      2@      3@      7@      =@      4@      >@      0@      A@      ;@      A@      <@      ?@      <@     �@@      >@      C@      F@      C@     �E@     �D@      L@      I@      I@      J@     �I@      K@      M@      J@     �M@      O@      V@     �R@     �R@      R@      S@     �S@      V@     @W@     �\@      Z@      \@      Y@     @_@      [@      a@     @b@     ``@     �`@     �a@      d@     �c@     �f@     `g@     �f@     �f@     �j@     `k@     �m@      k@     Pq@     �o@     �o@     �q@      u@     w@     �t@     pv@     �w@     @s@     0x@     �u@     {@      z@     0x@     �z@     �~@      @      @     ��@     �@     h�@     ��@     Ѓ@     ��@     ��@     8�@     p�@      �@     ��@     ؋@     ��@     `�@     p�@     \�@     ��@     ��@     `�@     ��@     ��@     ��@     \�@     L�@     �@     ��@     ��@     F�@     آ@     �@     ��@     Z�@     p�@     d�@     l�@     *�@     �@     ֭@     s�@     �@     K�@     �@     �@     B�@     E�@     ��@     ֹ@     �@     �@     L�@    �1�@    �,�@    �!�@     �@    ���@    �=�@    ���@    ���@    ���@     ��@    ���@     5�@    ���@     ��@    ���@     �@     t�@     ��@    ���@    �,�@    ��@    �T�@     M�@    ���@     ��@    �^�@     ��@    ��@     Խ@     ?�@     κ@     6�@     k�@     w�@     ��@      �@     k�@     Ϲ@     ߹@     %�@     u�@     ��@     ��@     ��@     �@     �@     �@     p�@     ܐ@     ��@      s@     �Q@      ,@      @      @       @      �?        
�
predictions*�	   `���   �ql�?     ί@!  4���.@)ף�XR"@2�
I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.�������6�]���>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��a�Ϭ(���(����vV�R9?��ڋ?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�������:�
              �?      &@      E@     @Y@      g@     �o@     @l@     �k@     �i@     �f@      a@     �[@     �U@     �Q@     @P@      S@      I@      H@      D@     �C@      =@      :@      :@      5@      2@      5@      ,@      (@      .@      ,@      "@      $@      "@      (@       @      @      @      @      @      @      @      @      @      @       @      @      @      @       @      �?      @      @      @      @      @      �?      �?               @      �?      �?              �?      �?               @              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?              �?      �?       @      @              �?              �?      @       @      �?      @       @      �?       @      �?      @      @      $@      �?      @      "@      @      @      "@      (@      &@      ,@      "@      (@      5@      0@      6@      8@      ?@      A@      :@     �A@     �B@      B@      D@      =@      >@     �C@     �@@      :@     �C@      G@      <@      E@     �C@     �D@     �E@      K@      G@     �E@      G@      D@     �F@     �G@     �D@     �E@     �D@      ?@      A@     �C@      0@      :@      4@      (@      4@      2@      $@       @      �?        � _�0      {��	T��l���A*�`

mean squared error׌:=

	r-squared �G=
�K
states*�K	   ��u�   @�E@    c2A!ߦ��z�@)b�� ��@2�%�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�̴�L�����/k��ڂ�\��$��%�f*��8ŜU|�x�_��y�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�%               @      @      (@     �V@     ��@     $�@     p�@     H�@     ,�@     П@     �@     1�@     �@     ��@     ��@     1�@     �@     ��@     d�@     I�@     ��@     ��@     K�@     �@     ��@     �@     �@     �@     ȼ@     ��@    ��@     +�@     ��@    �P�@     L�@     �@     ��@     ��@     ��@     ��@    ���@    ���@     ��@    ��@    @@�@    ��@     c�@     ��@    ���@     ��@     ]�@     |�@     ?�@    �,�@     ��@     ��@     ��@     ��@     ��@     ,�@     ��@     �@     a�@     ?�@     ޱ@     4�@     5�@     �@     v�@     ȩ@     ,�@     x�@     ��@     "�@     ��@     ��@     ��@     |�@     �@     �@     T�@     ��@     D�@     ĕ@     ,�@      �@     D�@     �@     �@     ��@     �@     x�@     �@     ��@     ��@     ��@     H�@     �@     �@     ��@     ��@     ��@     ��@     ��@     ��@     �~@     p�@     �|@     �}@     �x@     Px@      {@     �w@     �v@      y@     �u@     �u@     Ps@     �r@     Ps@     �p@      s@      n@     @q@     `m@      m@      k@     �g@     @j@      g@     `h@     `e@      g@     �f@     �c@     �d@     �d@      c@     �`@     �a@     �a@     ``@     �]@     �`@     �\@     @_@     �]@     �U@     �]@      Y@     @Y@     @U@      V@     �S@     @Q@     @U@     �T@     �M@      S@      L@     @Q@      U@     �M@      F@     �P@      K@     �J@      N@     �J@      D@      L@      I@     �@@      >@      >@      =@     �F@      A@      >@      @@      =@     �@@      ?@      =@      ;@      :@      6@      9@      :@      :@      2@     �@@      9@      2@      0@      9@      2@      (@      1@      0@      ,@      ,@      0@      4@      0@      $@      1@      $@      $@      ,@      &@      @      &@       @      &@      &@       @       @      @      "@      $@       @      @      @      @      @       @      @       @      "@      @      @      @      (@       @      @      @      @      @      @      @      @       @      @               @      @      @      @       @      @      @      @      @       @      @      @      @      @      �?      @      @      @      @       @       @              @      �?      �?      �?      �?      @       @      �?      @               @       @      @       @       @      @      @      @      �?      @       @       @       @      �?      @       @       @               @      @      @      �?       @     �Y@     �R@       @      �?       @               @              �?       @              �?       @      �?      @              �?              @              @               @              @              @      @      @       @      �?      @      @       @      @       @      @      @       @       @      �?      @       @      @      @      @      @      @      @      $@      @      @      @      @       @      @      @      @      $@      @      @      @      @      @      @      @      @      @      @      @      "@      @       @      @      @      "@      @      "@      &@       @       @      (@      "@      @       @      $@      $@      "@      ,@      @      &@       @      .@      (@      "@      1@      &@      (@      &@      ,@      0@      0@      4@      1@      (@      2@      1@      <@      2@      8@      9@     �@@      :@      ;@      9@      4@      ?@      D@      ?@     �B@      ?@      =@      C@      B@      A@      D@      A@      ?@     �J@     �G@      G@      E@     �G@      J@     �G@      Q@      O@      O@     �K@     �Q@     �R@      T@     �S@     @V@     �W@     @U@      W@     �V@     @[@     �Z@     �]@      \@     ``@      ^@     �]@      b@     @c@     ``@     �b@     `c@     �e@      f@     �f@      i@     �d@     �g@     `i@     �j@     �i@     @j@     �m@      n@     pp@     �q@     �r@     �s@     �s@     �r@     pz@     �v@     pv@     �y@      w@     0y@     z@     �{@     @|@     �{@     ~@     0�@     ؁@     p�@     ��@     H�@     ��@     X�@     ��@     @�@     @�@     �@     Њ@      �@     ��@     �@     h�@     `�@     ��@     ��@     ܑ@     8�@     ��@     ��@     L�@     p�@     T�@     ��@     ��@     p�@     ��@     H�@     ��@     ��@     �@     ~�@     P�@     ҥ@     Ц@     ��@     «@     ҩ@     |�@     ح@     �@     a�@     m�@     ��@     0�@     ��@     ն@     ��@     w�@     ��@      �@     m�@     \�@     ��@     Q�@     ��@    �#�@     -�@    ���@    ��@     ��@     ��@     ��@     ��@     P�@    ��@     ��@    �B�@     7�@    ���@     �@    ���@     ��@    ���@     7�@    ���@     p�@     ��@     j�@     ��@     /�@     Ϲ@     !�@     L�@     �@     	�@     ��@     7�@     Z�@     �@     �@     ι@     /�@     ��@     ��@     ��@     }�@     u�@     d�@     `�@     `�@     p|@     �g@     �H@      ,@       @      @      �?        
�
predictions*�	   ����   ���?     ί@!  ���B�)U�HQ?z#@2�
�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��.����ڋ������6�]���1��a˲���[��5�"�g���0�6�/n���ߊ4F��>})�l a�>�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?�������:�
              �?      2@      R@     `e@     �m@     �l@     �l@     `j@      i@      c@      b@     �b@      _@     @W@     @Z@     �R@     @S@     �R@     �P@     �J@     �B@      A@      >@      D@     �A@      2@      1@      4@      1@      6@      4@      .@      *@      $@      "@       @      *@      @      "@      @      @      @      @      @      @      @      @      @      @       @      @      @       @      �?              �?               @       @       @              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?               @      @      @               @       @       @       @              �?       @       @      �?      @      @       @      @      @      @      @      @      @      @      $@       @      (@      &@      (@      *@      ,@      .@      3@      :@      2@      3@      3@      3@      <@      8@      1@     �@@      8@      4@      @@      .@      :@      3@      ;@      9@      ;@      ?@      ?@      >@      @@     �B@     �C@      ?@      <@      :@      8@      ?@      >@      3@      .@      1@      7@      (@      $@      (@      @      @      "@      &@      @       @      @       @      �?        ��V��0      z�{	�۱l���A*�a

mean squared error�9=

	r-squaredЊ]=
�L
states*�L	   �;��    c�@    c2A!y�A�ۧ�)��`�қ�@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%               @       @       @      4@     `j@     Ј@     ��@     ��@      �@     ��@     �@     ��@     5�@     W�@    ���@     {�@     ��@     S�@     �@     O�@     �@     �@     �@     7�@     ��@     ��@     `�@     E�@     U�@     ��@     ��@     ��@     W�@    �k�@     
�@     v�@     ��@     ��@    ���@    ��@     e�@    ���@    �@�@    ���@    �F�@    ���@    ���@     ��@    �I�@    ���@     ��@    ���@     v�@     ��@     �@     ��@     9�@     ��@     ��@      �@     �@     9�@     Q�@     �@     <�@     ��@     ��@     x�@     ح@     v�@     �@     `�@     �@     .�@     �@     ��@     �@      �@     ܞ@     l�@     ܚ@     ��@     Ę@     ��@     ��@     ܔ@     ��@     ��@     @�@     0�@     L�@     �@     ��@     ��@     �@     P�@     (�@      �@     (�@     І@     �@     ��@     ��@     P�@     ��@     ��@     0�@     �@     �~@     p}@     �z@     ��@     {@     �y@     �v@     �v@     �w@     �t@     @t@     �t@     �s@     �q@     pq@     �p@     Pq@     @m@     �p@     �q@      m@     `h@     �l@      h@     �i@     �i@      h@     �h@     �e@     �e@      g@     `c@      b@     �c@     �b@     `c@     ``@      _@     �b@     �^@      ^@      ]@      [@      ]@     �S@     �W@     @Z@     �X@     @W@     �X@     �S@      U@      T@     @R@     �Q@     @P@      U@      J@     �Q@     �J@      I@     �F@     �L@      G@     �J@      I@     �E@     �C@      F@     �D@      B@      A@      A@      <@     �@@      1@      B@      A@     �@@      7@     �C@      :@      1@      <@      5@      B@      5@      <@      1@      3@      ,@      7@      &@      0@      ,@      ;@      2@      *@      1@      3@      $@      &@      @      *@      0@      @      .@      $@      (@      "@      @      @      @      @      &@      @      &@      "@       @      @      @      @      "@      $@      @       @      @      @      @      @      @      @      @      @      @      @      @       @      @       @      @      "@      @      @      @      @      @       @      �?      @       @      @      @      @      @      @      @      @      @       @      @       @      @       @      @      @      @              @      �?      @       @      @       @      @              @       @       @      �?      @       @      �?              @      @      �?      �?      @       @       @      �?      �?     �\@     @[@               @      �?       @      �?      �?      @              �?       @       @      @      �?       @      @      �?      @      @      @      �?      @       @      @      @      @      �?      @      �?       @      �?      @      @      @      @      @      �?      �?       @      @      @      @      @      "@      @      @      @      @      @      "@      �?      $@      @       @      @      @       @      @      @       @      @      @      @      @       @      "@      "@      "@      @       @      $@      @      "@      @      (@      "@      @      @      .@      ,@      $@      *@      .@      $@      ,@      &@      "@      $@       @       @      ,@      (@      .@      (@      2@      *@      2@      5@      .@      4@      6@      6@      1@      7@      <@      @@      6@      <@      ?@      5@      ;@      ?@     �@@      @@     �D@      >@      A@      B@     �D@      B@      D@      A@     �G@      D@     �G@     �J@     �H@      O@     �H@      N@      I@      P@      R@     �O@     �L@     �O@     @R@     �T@     �U@     �W@     �S@      S@     @]@     �X@     @Z@      X@     �[@     @`@     �b@     �b@     �`@     @c@      c@     @e@     �d@     �d@     �e@     �f@      j@     �g@     �l@     �i@     `m@     �k@     �l@     �o@     `q@     �p@     �q@     �q@     0r@     �s@     �s@      u@     w@     0u@      w@     `v@      w@     �v@     �{@     �z@     �@     `@     0|@     �@      @     ؀@     h�@     ��@     ��@     @�@     �@      �@     P�@     ��@     `�@     0�@     �@     ��@     ��@     ؎@     h�@     ��@     x�@     `�@     �@     ��@     $�@     d�@     ��@     ̔@     Ȕ@     ȕ@     ,�@     ��@     ��@     D�@     ��@     <�@     .�@     �@     �@     ¢@     ��@     ��@     ��@     �@     ��@     ,�@     ��@     ��@     ֯@     ��@     �@     Ʋ@     ٴ@     �@     ��@     [�@     ��@     k�@     D�@     (�@     ��@    ���@    ���@    ���@     7�@     8�@    ���@    �n�@    ���@    ��@    ���@     `�@    ���@    ���@     X�@     ��@     ��@     ��@     v�@    ���@     ��@    ���@    �U�@     ��@    �V�@     ��@     ��@    ��@    �[�@     ��@     ��@     _�@     F�@     ,�@     _�@     ַ@     B�@     $�@     �@     �@     ,�@     ׸@     �@     o�@     �@     ն@     ȶ@     ��@     �@     X�@     `w@     �g@     �P@      5@      $@      @      @      �?        
�
predictions*�	   �ᓳ�    ���?     ί@!  �/BM@)�-��2@2�
�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$���ڋ��vV�R9��T7����5�i}1�6�]���1��a˲��ߊ4F��>})�l a�>I��P=�>��Zr[v�>>�?�s��>�FF�G ?����?f�ʜ�7
?�T7��?�vV�R9?��ڋ?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�������:�
               @      <@     �V@     @c@      k@     �e@      h@      c@     @_@      W@     @V@      N@      O@     �I@     �C@      @@     �B@      G@      =@      <@      9@      4@      9@      8@      0@      0@      2@      .@      ,@      (@       @      &@      @      @      @      @       @      @      @      @       @       @      @      @      @              @      �?      �?      �?      @      @      @              �?              �?               @       @       @              @              �?              �?              �?              �?              �?              �?              �?              �?      �?               @      �?       @      �?               @      �?      �?              �?              @      @      �?      @      @      @      @      @       @      @      "@      .@      @      @       @      ,@      1@      0@      .@      4@      9@      7@      9@      >@      :@      B@      ?@      A@     �A@      A@      ?@      C@      <@      A@      F@     �A@     �J@     �N@     �H@      H@     �P@     �L@     �H@     �F@      S@      Q@     �K@      R@     �S@     @R@     �R@     �I@     �O@     �P@      L@      F@     �@@      @@      ;@      6@      2@      *@      @      $@       @      @      @      �?        ��1      "���	k��l���A*�b

mean squared error�6=

	r-squaredЬ�=
�L
states*�L	   �>��   ���@    c2A!f��
��)������@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              �?      �?      @      ;@     �p@     �@     ��@     �@     ��@     4�@     ȡ@     )�@     ��@     �@     ?�@     �@     ��@     մ@     t�@     ٴ@     "�@     R�@     ��@     ��@     v�@     Ҷ@     ��@     ۷@     9�@     8�@     �@     ��@     8�@     �@    ��@    ���@     ��@     ��@     P�@     ��@    ���@     ��@    @��@    ���@     ��@    ���@    �/�@    �,�@     ��@    ���@     Q�@     ��@     �@     y�@     ��@     ��@    ���@     �@     
�@     ��@     ��@     5�@     ��@     o�@     h�@     ��@     �@     ǰ@     �@     &�@     "�@     �@     L�@     ��@     >�@     0�@     z�@     �@     �@     h�@     ��@     (�@     L�@     ��@     ��@     ��@     ��@     $�@     p�@     8�@     t�@     ��@     ��@     �@     8�@     ȋ@     8�@     ��@     p�@     P�@     ��@     ��@     x�@     `�@     �@     8�@     ��@     p�@     ��@     ؄@     �~@     �z@      }@     {@     P|@     �y@     �x@     �z@     @�@     �v@     pt@     �t@     @s@      t@     `r@     t@     �q@     pq@     �q@     �o@     �p@      i@     `o@     �l@     �g@     �h@     �e@     �i@      h@     �e@     �g@      e@     �e@     �d@     �d@     �b@     �b@     `b@      e@     @]@     �_@     �`@      `@     @[@      Y@      V@     �U@      Y@      W@     @V@     �R@     �V@     @W@     �Q@     �U@      T@     �T@     �P@     �O@     �Q@     �J@      Q@      K@      H@     �K@      H@      E@      E@     @P@      A@      B@     �A@      A@     �G@      A@     �E@      B@      A@     �A@      8@      ;@      9@      ?@      <@      <@      1@      <@      ?@      3@      0@      5@      4@      6@      7@      *@      .@      $@      .@      2@      &@      .@      ,@      *@      1@      5@      &@       @      *@      ,@      @      @      (@      @      "@      @      @      $@      "@      @       @       @      @      "@      $@       @      @      "@      "@      "@      @      @      @      @      @      @      @      @      @      �?      @      @      @       @      @      @      @      @      "@      @      @      @      @       @       @      @      @      @      @      @      @      @      @       @      @      @      @       @      @      @      �?      @      �?              @       @      �?      @      @               @      @      �?      �?       @      @              �?      �?      �?              @     �`@     @`@      �?              �?       @      �?      @      @      @      @      @       @      �?      @       @       @      @      @               @      @      @      �?      @      �?      @      @      �?      @      @      @       @      @      @      @      @       @      �?      @      @      @      @       @      @       @      @      @      @      @       @      @      @      @      @      "@      @       @      @      @      @      @      "@       @      &@       @       @       @       @      &@      "@      .@      "@      @       @      5@      $@      "@      (@      *@      $@      1@      2@      0@      &@      (@      (@      2@      4@      0@      4@      1@      6@      3@      0@      0@      3@      2@      1@      @@      =@      2@      ;@      5@      :@      ;@      @@      >@      7@      B@     �F@      A@     �D@      C@      C@      @@     �H@      C@     �E@     �G@     �H@     �N@     �D@     �O@     �K@      P@      I@      O@      P@     �P@      N@     �Q@     �U@     �T@     �S@     �U@     �W@     �V@     �X@     �X@     @]@      `@     �[@     @a@     �]@     �c@     �b@     @a@     �c@     �b@      f@     @d@      c@      g@      g@     �h@      h@     �j@      m@     �l@     �i@     @k@     Pp@     �p@     `o@     `p@     @p@     �q@     �p@     Ps@      u@     s@     �u@     �u@     �u@     �u@     x@     `v@     �w@      {@     �z@     y@      |@     P|@     X�@     �@     @�@     0�@     ��@     0�@     ��@     P�@     �@     �@     ��@     x�@     ��@     x�@     �@     ��@     Љ@     ��@     H�@     ؍@     �@     ��@     ,�@     d�@     ԑ@     �@     ��@     ��@     \�@     ��@     �@     �@     Ԙ@     ș@     ��@     ܞ@     |�@     *�@     ��@     ޠ@     P�@     �@     d�@     �@      �@     ��@      �@     .�@     B�@     ��@     �@     �@     ��@     ��@     �@     ��@     `�@     ·@     ��@     �@     ��@     ��@     ��@     ��@    �~�@    ���@    �9�@     ^�@    ���@    ���@     ��@    �	�@    ���@     ��@    ���@    �X�@    �n�@    ���@     ��@     ��@     z�@    ���@     ��@     �@     ��@     Z�@    ���@    �1�@     	�@     ѹ@     -�@     ��@     �@     Z�@     `�@     ��@     õ@     Ҵ@     ��@     G�@     �@     2�@     3�@     ߸@     ��@     ۹@     &�@     0�@     ,�@     ��@     ��@      s@     �b@     �L@      9@      .@       @      �?      @        
�
predictions*�	   ��ұ�   `���?     ί@!  :�Fd>�)�b�@2�
��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9���vV�R9��T7���x?�x��>h�'��1��a˲���[��O�ʗ�����Zr[v����Zr[v�>O�ʗ��>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?W�i�b�?��Z%��?�������:�
              �?      @      (@     �L@     �\@      f@      h@     @k@      j@     �e@     �`@     �c@     ``@      \@      [@      [@     �X@     �U@     @X@     �W@     �Q@      R@     �S@     @R@      H@     �H@     �D@      ?@      4@      <@      8@      3@      :@      3@      "@      *@      "@      &@      1@      $@      (@      @      @      @      �?      @      @      @      @      @      @       @      @      @       @      �?               @              �?              �?      �?              @              �?              �?              �?              �?              �?              �?               @              �?              �?      �?               @      �?      �?              �?              @      �?      @       @       @      @      �?      @      @      @      @      "@      �?      �?      @      "@      "@      $@      @       @      @      &@      &@      .@       @      *@      ,@      1@      ,@      2@      .@      2@      ;@      7@      3@      <@      ;@      7@      7@      ;@      8@      =@      9@     �@@      A@      8@      <@      ;@      =@      ?@      <@      9@      7@     �@@      *@      2@      2@      .@      2@      1@      (@      0@      (@      @      "@      $@      $@      @       @      @      �?      �?              �?        (��@R1      #}�a	u�l���A	*�b

mean squared error��2=

	r-squared\�=
�L
states*�L	    �{�   �A�@    c2A!<� ����)_M�)�@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%              �?       @      @     �]@     �@     (�@     ��@     P�@     �@     (�@     ��@     �@     I�@     �@     	�@     �@     ��@     �@     ��@     (�@     ��@     x�@     )�@     �@     3�@     ��@     K�@     ٷ@     ƹ@     &�@     "�@     �@     ��@    ���@      �@     ��@    �_�@     ��@    �S�@     w�@    ��@    ���@    ���@     ��@    �N�@    ���@     ;�@    �}�@     �@     ��@     �@     �@     ��@     )�@     ��@    ���@     ��@     ��@     ��@     �@     w�@     ��@     ۹@     �@     1�@     ʹ@     *�@     -�@     
�@     �@     ��@     :�@     P�@     ��@     >�@     �@     �@     \�@     :�@     l�@     R�@     �@     Ĝ@     ��@     ��@     \�@     x�@     �@     �@     ؒ@     (�@     �@     ��@     �@     ��@     8�@     h�@     ��@     �@     ��@     ��@     �@     X�@     �@     ��@      �@     `�@     �@     ��@     ��@     �@     @~@     ��@     P}@      }@     @|@     �y@     �|@     �x@     �y@     0x@     �u@     �v@     �u@     �s@     pt@     �w@     �x@     r@     �q@     @q@     �o@     �p@     `o@     `n@     `k@     �h@     �j@     @k@     �g@     @i@      h@     `g@     �e@     �d@     �c@     `e@     �b@      d@     @e@     �a@     �b@     �a@      ^@     �]@     @^@     @_@     �^@     �X@     @Y@     @Y@      V@     @U@     @V@     @Y@     �S@     �V@     �U@     �Q@     �N@     �T@     �P@      O@     �P@      N@      V@     �J@      C@      N@      D@      G@      K@      F@      H@     �F@      B@     �F@     �B@      A@      G@     �@@      A@      E@      ;@     �B@      D@      =@      >@      B@      <@      <@      9@      4@      @@      8@     �@@      0@      1@      B@      2@      5@      ,@      2@      (@      .@      .@      5@      7@      2@      ,@       @      &@      .@      (@      &@      $@      @      @      *@      1@      @      "@      "@       @      5@      @      @      ,@      @      @      @      @      @      "@      @      (@      @       @      @      @      @       @      @      @      @      @      @      "@      @      &@      @      @      @      "@      @      @      �?      @      @      @      @       @      @      @      @      @      @      @      @       @      @      @      @       @      @       @       @       @       @      @       @       @      @      �?      @      @              @              �?      @     �f@     �]@      @       @       @       @      @      @      �?       @      �?      @      @      @       @      @      @      @       @      @      �?      @      @      @       @       @       @      @      @      @      @      "@       @      @      @       @      @      @      �?      �?      @      @      @      @      @       @      "@      @       @       @      @      @      @      @       @      @      &@      @       @      @      @      &@      $@      (@      @      ,@      @      &@      $@      *@      &@      $@      @       @      "@      $@      2@      (@      1@      4@      .@      ,@      1@      1@      *@      ,@      0@      *@      ;@      .@      :@      8@      :@      2@      5@      9@      ;@     �@@      ;@      :@      =@      =@      >@     �A@      4@     �A@      A@      >@     �B@     �C@      I@     �A@     �H@     �E@      B@      J@     �K@      I@     �D@     �G@     �F@      J@     �O@     �Q@     �R@     �T@     @U@     �V@     @V@     �X@     �Q@     �X@     �X@     �]@     �]@     @\@     �^@      Z@     �`@     �[@      \@     @a@      b@      b@     �`@     �b@     �c@     �d@     �f@     �d@     �g@     `f@     �h@     `h@     �h@     �h@      k@      l@     `i@     @k@     �n@     p@     pq@     @p@     �q@      r@     �q@     t@      r@     �r@     �t@     t@      v@     v@     �w@     �v@     pv@     �x@     �x@     �y@     }@     �@     ȁ@     P@     ��@      �@     @�@     H�@     x�@     P�@      �@     ��@     �@     ��@     Ȅ@     h�@     ȇ@     h�@     ��@     0�@      �@     ��@     ��@     h�@     �@      �@     0�@     ��@     ��@     ��@     �@     ,�@     t�@     �@     ��@     ԙ@     ��@     x�@     |�@     �@     :�@     ̡@     .�@     �@     �@     J�@     ^�@     �@     ��@     ��@     ȫ@     ��@     ��@     ��@     ��@     ��@     a�@     �@     Է@     �@     /�@     ̻@     ��@    ���@     ��@    ���@    �i�@    �q�@     o�@     ��@     ��@    �7�@    ���@     ��@     ��@    ���@    ���@     G�@     ��@     ��@    ���@    �w�@     T�@    �M�@    ���@     #�@    �y�@    �N�@     ��@     ��@    �#�@     Ͼ@     �@     �@     ��@     `�@     u�@     !�@     k�@     |�@     X�@     �@     ��@     k�@     ��@     h�@     �@     ��@     e�@     �@     �@     s�@     t�@     X�@     H�@     0x@     �g@      T@      ?@      4@      "@      @      @        
�
predictions*�	   �����   �2`�?     ί@!  ����R@);�$�B
2@2�
��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲��FF�G �>�?�s���R%�����>�u��gr�>��[�?1��a˲?6�]��?>h�'�?x?�x�?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?�������:�
              @      :@     �Q@     �Z@     �[@      ^@     �Z@     �Z@     �Z@      Y@     �Y@      T@     �L@      K@      J@     �Q@     �C@      D@      C@     �A@     �B@      @@     �A@      *@      <@      1@      &@      ,@      *@      $@      @      "@      @      "@      ,@      "@      @      "@      @      @      @      @      @      @       @      @      @      @      @      @              @              �?              �?      @       @              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?              �?              �?       @      �?      �?              �?              �?              @       @      �?      @      �?       @       @      @      �?      @      @      @      @      @      @      @      @      @      @       @      "@      "@      *@      *@      .@      7@      &@      <@      8@      5@      8@      3@      8@      ;@      ?@      >@     �B@      B@     �D@      H@     �O@     �L@     �I@      S@     �N@     �I@      U@     �Y@     �Z@      U@     @T@      N@      O@      O@      N@      P@     �I@     �L@     �L@      G@     �@@      A@      I@     �B@      G@      A@      1@      =@      9@      4@      1@      ,@      "@      @       @      @      �?       @      �?      �?         ��1      "���	� m���A
*�b

mean squared errormb-=

	r-squared�s�=
�L
states*�L	   �-�   @�$@    c2A!2^�����)�4�����@2�%h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�%              �?      &@      r@     ��@     h�@     ؂@     `~@     �|@     Є@     ��@     ��@     r�@     ��@     ��@     /�@     ��@     �@     *�@     ��@     x�@     �@     ܴ@     ��@     ��@     �@     ��@     ۵@     ʷ@     �@     �@     f�@    �f�@    ���@     ��@    �s�@    ���@    ���@     ��@     ��@    ���@     ��@     ��@    ���@     ��@    �B�@    ���@     ��@     ��@    ���@     ��@     W�@    �S�@    ���@    ���@    ���@    ��@     �@    ���@     ��@     z�@     '�@     �@     F�@     µ@     �@     ��@     F�@     �@     ~�@     ��@     �@     ��@     �@     ��@     ��@     �@     B�@     f�@     �@     P�@     ��@     ��@     ��@     <�@     ,�@     h�@     ��@     ��@     h�@     ̒@     �@     ��@      �@     ��@     @�@     P�@     ��@     ؊@     ��@     ��@     0�@     P�@     �@     h�@     Ѕ@     ��@      �@     ��@      �@     ؁@     `�@     X�@     ��@     X�@     `|@     0{@     pz@      |@     �y@     `y@     @y@     �y@     �w@     `v@     �w@     �v@     �s@     @t@     @s@     `r@     `t@     `w@      t@     @o@     `o@     �p@     @j@     �n@     �j@      l@     @i@      k@     �i@     @f@      h@      e@     @d@      g@     �c@      b@      _@     @c@     @b@     @a@      a@     @`@     @_@     �\@     �Z@     �^@      Z@     �V@     �W@     @V@     �Y@      Y@     �V@     @T@      Z@     �S@     @S@     �U@     �Q@     �R@     �R@     �O@     �P@      Q@      O@     �K@      H@     �N@      G@      D@      H@     �G@     �D@      C@      G@      F@     �C@     �F@     �F@     �A@      @@      A@      A@      D@      >@     �C@      7@      C@     �D@      8@      7@      =@      6@      2@      3@      4@      4@      4@      3@      .@      0@      3@      .@      1@      7@      ,@      4@      .@      $@      .@      *@      $@      "@      *@      ,@      1@      @      ,@      ,@      &@      ,@      ,@      (@      "@      @      @      *@      @       @      @      &@      @      (@      @      &@       @       @      @      @       @       @       @       @       @      @      @      �?      @      @      @      @      @      @      @      @      @      @       @      @      �?       @      @      @      @      @      @       @      @       @      @      @      @      @      �?      @      �?      @       @      �?      �?      @      @       @      @      �?      @      @     �d@     �h@      @      @       @      @      @              �?      �?      @      @       @      @      @      @      @       @      @      @       @      @      @      @      @      @       @       @      @      "@      @      @       @      @      @       @      $@      @       @      "@      "@      "@      @      @      ,@      $@      $@       @      @      $@      @       @      "@      "@      @      @      "@      "@      2@      @      (@      ,@      $@      *@      "@      &@      &@      $@      0@      $@      ,@       @      *@      ,@      *@      2@      ,@      (@      2@      2@      *@      4@      4@      4@      <@      .@      =@      =@      0@      6@      :@      =@     �@@      7@      2@      ?@      =@      @@     �B@      A@      6@      C@      <@      C@      =@      A@      ?@      A@      K@     �D@      C@      E@     �K@     �O@      K@      G@     �D@      M@     �P@      G@     �N@     �Q@     �T@     @R@     �Q@     �T@     �T@     �T@     �Y@     �[@     @Z@     �[@      Z@     �V@     �Y@     �Z@     �Y@     @`@      _@     �\@     �`@     @_@     @a@     �a@     @a@      a@     @b@     �d@     �c@     `e@     �e@      f@     @h@      j@     @k@     �l@     �i@     �m@     @l@     `k@     @m@     �n@     �p@      o@     `p@     �q@     `q@     �r@     Ps@     @s@     �u@     v@     �v@     pv@     �u@     x@     `y@     �y@     px@     �z@     @|@     p}@     �}@     �@     �}@     @@     ��@     ��@      �@     ��@     Ђ@     P�@     ��@     ȇ@     �@     8�@     0�@     ؆@     ��@     ��@     h�@     X�@     @�@     8�@     `�@     ��@     ��@     �@     H�@     |�@     ��@     D�@     �@     ��@     l�@     ��@     d�@     L�@     ��@     ��@     (�@     ԟ@     ~�@     ��@     ^�@     h�@     ��@     ��@     ��@     j�@     @�@     ��@     �@     ��@     İ@     �@     �@     Q�@     ��@     5�@     ȶ@     ��@     ��@     y�@     ��@     �@    ���@    �W�@    ���@    ���@    ���@    �"�@     ��@    ���@    ���@    ��@    �+�@    �X�@    ���@    ���@    �N�@    � �@    �)�@    �5�@     j�@    ���@    �"�@     ��@     ��@    �Q�@    ���@     4�@    ��@     �@     �@     ��@     ��@     ��@     $�@     }�@     8�@     W�@     �@     ��@     
�@     ��@     ȸ@     s�@     �@     ��@     �@     ��@     L�@     ��@     8�@     �v@     �p@     �V@      E@      7@      $@      &@       @      �?        
�
predictions*�	   �����    0��?     ί@!  +X$�T�)�Z����,@2�
��(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7�����d�r?�5�i}1?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?������?�iZ�?�������:�
              �?      "@      B@      T@     @e@     �j@     @h@     �h@     @h@     �i@      i@     �i@     �f@     �c@     ``@      _@     @[@     �T@      S@     @S@     �O@      Q@     �P@     �I@      F@      I@     �D@      A@      :@      4@      3@      3@      4@      *@      .@      $@      .@      *@      &@      @       @      @      @      $@       @      @      �?      @      @      @       @              �?       @      �?       @      �?      @       @              �?               @       @              �?      �?              �?              �?      �?              �?               @              �?              �?       @              �?       @      @      �?              �?       @      @      �?       @      @      @      �?      @      @      @       @      @       @      @      @      @      *@      @              @      @       @      *@      "@      "@      2@      @      &@      2@      .@      3@      4@      5@      ,@      ,@      3@      5@      .@      .@      $@      (@      3@      4@      6@      4@      *@      9@      (@      2@      $@      2@      6@      ,@      6@      ,@      3@      &@      .@      "@      @      0@      @       @      �?      @      @      @      @      �?      @      �?       @              �?              �?        ��&1      "���	�qm���A*�b

mean squared errorl�(=

	r-squared�g>
�L
states*�L	   `˾�   �#�@    c2A!��%�eר�)#S�cGE�@2�%�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%             �_@     �@     p�@     ��@      �@     �}@     P�@     h�@     $�@     |�@     Ҹ@     /�@     Ǵ@     �@     ��@     �@     ��@      �@     �@     ��@     �@     ��@     �@     ʶ@     ��@     c�@     �@     ��@     ;�@     E�@     s�@     ��@     ��@    �
�@    �=�@    �]�@    ���@    �m�@    �g�@     ��@     ��@    ���@    ���@     ��@    �G�@    �
�@     ��@     0�@     ��@    �M�@    ���@     ��@    ���@     �@     �@     ��@    �0�@    ���@    �W�@     �@     k�@     &�@     T�@     �@     ״@     ��@     ��@     x�@     ��@     έ@     \�@     �@     H�@     ��@     &�@     ��@     ��@     �@     j�@     D�@     ��@     ��@     �@     ��@     ��@     ��@     ��@     l�@     ��@     ��@     H�@     ��@     ��@     |�@     H�@     \�@     �@     ��@     �@     ��@     Ȉ@     �@     ��@     ��@     h�@     ��@     P�@     �@     �@      �@     P�@      @     P~@     (�@     �}@      z@     {@     �y@     Py@     Pz@     Pw@     �u@     �u@     �v@     `t@      u@     u@     �s@     �q@     �t@     �p@     Pr@     0p@     �p@     @t@      r@     �k@      o@      l@     �h@     �g@     �d@      g@     �h@     �f@      d@     `f@     `c@     @c@     �`@     `a@     �`@     �a@      a@     �d@     @]@     @`@     �`@     �\@      `@      [@     �^@      Y@      W@      U@     @V@     �Z@     @T@      R@      S@     �U@     �R@     �T@     �R@     �P@     �R@      I@     �P@     �Q@      C@      M@      P@     �E@      I@      Q@      J@     �G@     �L@      J@      E@      B@     �G@     �E@      C@     �@@      8@      >@     �@@     �D@      =@      =@      9@     �@@      8@      :@      >@      ?@      ;@      7@      >@      5@      2@      8@      0@      .@      <@      3@      <@      ,@      (@      ,@      ,@      5@      ,@      .@      &@      7@      .@      &@      *@      ,@      "@      2@      *@      (@      &@      &@      &@      *@      ,@      @      @      @      "@      @      $@      "@      @      @      "@      *@      @      "@      @      "@      @       @       @      "@       @      @      @      @      @      @       @       @      @      @       @      @      @      @       @      @      @       @      @      @      @      @      @      @      @      @      @      @              @      @      �?      @       @      @      @      �?      @      "@       @      @     �h@     �h@      @       @      �?      @       @      @      �?      @      @      �?      @      @      @      @      @      @      @      @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @      "@       @      "@       @      @      (@      "@      @      @      @      @      *@      &@      ,@      ,@       @      &@       @      (@      $@      @      &@      2@       @      *@      &@      *@      3@      ,@       @      1@      .@      &@      (@      0@      0@      *@      2@      &@      5@      *@      9@      5@      7@      1@      4@      5@      :@      7@      :@      7@      7@      6@      @@      <@      B@      ?@      <@      :@     �A@     �C@     �G@     �A@     �@@      D@     �@@      B@      D@     �B@     �J@     �J@     �D@     �F@     �K@      H@     �H@     �N@      P@     �T@     @R@     �Q@     �P@      S@     @W@     �Q@      U@     �U@     �X@      X@     �W@     �[@     �\@     �Z@      Z@      ]@     �`@     @]@     �\@      `@     �`@      \@     @a@      b@     �a@     `b@      b@      a@      e@      d@     @g@     �d@     `h@     �f@     �g@     �i@     �m@     �k@     �l@     `l@     �m@      l@     q@     �o@      p@     0s@     `t@     Pr@     Pt@     �q@     �s@     Pr@     �u@     `w@     u@     �w@     �w@     `y@     �x@      {@     �z@      {@     �~@      �@      �@     X�@     �@     �@     p�@     ��@     ��@     ��@     ��@     Ȅ@     ��@      �@     ��@     �@     ��@     �@     Љ@     ��@     @�@     ��@     Ў@     ��@     �@     �@     �@     x�@     ��@     ��@     0�@     D�@     ��@     ��@     <�@     ��@     �@     8�@     n�@     l�@     Ρ@     ��@     &�@     ��@     B�@     ڦ@     ��@     N�@     �@     ��@     ޭ@     �@     ��@     `�@     ��@     w�@     ��@     ��@     ��@     ��@     6�@     s�@     ��@     <�@     ��@    �S�@    ���@     �@    ���@    ���@    ���@    ��@    ��@    ���@     ��@    �'�@    ��@    ��@     ��@     ��@     ;�@     �@     %�@     ��@     ��@     �@    ���@    ���@    ���@     ��@     o�@     ��@     ?�@     �@     ��@     �@     ��@     B�@     !�@     j�@     F�@     �@     ��@     K�@     �@     ޷@     5�@     �@     ]�@     �@     ��@     ��@     H�@     h�@      w@     @p@     �_@     �E@      6@      8@      "@      @        
�
predictions*�	   �mb��   @;��?     ί@!  ��pR@)�� ;B@2�
��(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��5�i}1���d�r�E��a�W�>�ѩ�-�>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�P�1���?3?��|�?�������:�
              �?      @      .@      ?@     �P@     �b@      e@     �g@     `e@     @c@     ``@     �^@     @R@     �V@     �Q@     �E@     �K@     �D@      K@     �D@      E@      A@      ?@      :@      ;@      3@      8@      <@      ,@      1@      .@       @      *@      @       @      &@      0@      @      @      $@      @      @      @       @      @      @      @      @       @      @      @      @      @       @      �?               @       @              �?               @      �?       @              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?       @              @              @       @      @       @       @      �?      @      �?      �?      @      @      @      �?      @       @      @      @      @       @      $@      @      ,@      *@      ,@      ,@      $@      (@      1@      8@      7@      8@      6@      3@     �@@      B@      D@      D@      F@     �N@      K@     �O@     @Q@     �Q@      M@     �P@      F@      P@     �H@      H@      J@     �N@      O@      G@      D@      L@      J@     �B@      F@      ?@      E@      C@      E@      ;@      E@     �@@      :@      ?@      *@      ,@      $@      &@      *@      @      @      �?      �?       @      @              �?        v�0      &ˣ	�@5m���A*�a

mean squared error��%=

	r-squared�]>
�L
states*�L	    t�   @�q@    c2A!.����]��)������@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �?     �W@     ��@     ��@     �@      {@     Py@     �z@      �@     0�@     ̗@     L�@     ��@     �@     ��@     `�@     e�@     �@     ,�@     h�@     8�@     �@     ��@     b�@     �@     �@     +�@     �@     ��@     :�@     �@     շ@     [�@     ۾@     �@     ��@     s�@    �U�@    ���@    ���@    �(�@     ��@     ��@    �Y�@    ���@    ���@    ���@    �;�@    ���@     ��@    ���@     ��@     ��@    �+�@     ��@     
�@    ���@    �;�@    ���@     f�@     ӿ@     ��@     ��@     ?�@     �@     �@     ��@     �@     �@     N�@     ��@     z�@     ��@     ڦ@     ��@     r�@     ��@     ^�@     �@     ��@     4�@     Ԝ@     ؚ@     �@     ��@     ��@     ĕ@     X�@     �@     ̓@     �@     ؐ@     ��@     ��@     �@     ��@     ��@     �@     ؈@     ��@     `�@     ��@     ȅ@     ��@     Ѓ@     8�@     �@     ��@     ��@      �@     ��@     �~@     �@      ~@     p|@      }@     �|@     �y@     Px@     py@     @x@      v@     `u@     Pv@     0v@     �u@      s@     @s@     �r@     Pq@     0q@     �p@     �q@     �n@     �o@      o@      o@     �l@      m@     �u@     �m@     `i@     �e@     �e@      h@      d@     �e@     �d@     @d@      d@     @b@     ``@      a@     @^@     �\@     �b@     �[@     �Z@     @b@      ^@     @]@     @W@      [@      ^@     �Y@      [@     �Q@     �X@     @R@     @R@     @Q@      O@      U@      T@     �Q@     �T@     �Q@     @Q@     �Q@     @R@     �N@      K@     @P@     �Q@     �P@      N@     �K@      H@     �D@     �G@      C@     �L@     �H@      E@     �A@     �D@      @@      C@      A@     �B@      6@     �H@      A@      =@      A@      @@      7@      ;@      ;@      8@      4@      >@      3@      0@      A@      <@      1@      2@      5@      0@      1@      .@      4@      3@      0@      .@      3@      5@      $@      $@      $@      1@      &@      *@      (@      $@      2@      *@      0@      .@      1@      (@       @      "@       @       @      *@      @      $@      "@      *@      @       @      @      "@      "@      @      @      @      *@      "@      @              @      @      @      @      $@      @       @      @       @      @      @      @       @      @      @      @      @      @       @      @      @      "@      @       @      @       @       @      @      @      @      @      @      @      @      @      �?      �?      @      �?     �h@     �j@      @       @      @      @      @       @      @      "@       @       @      @      @       @      @      @      @      @       @      @      @      "@      @      @      @      @       @       @      @      @      @       @      @      $@      @      &@      @      "@      @       @       @      @      @      @      @      $@      @      $@      $@      &@      @      ,@      @      "@       @      @      &@      @      &@      ,@      "@      ,@      @      *@      ,@      &@      *@      2@      0@      ,@      *@      0@      2@      0@      2@      6@      8@      2@      ;@      3@      0@      A@      5@      ;@      7@      8@      >@      =@      4@      ;@      5@      1@      4@      B@      <@      ;@      B@      >@      9@      C@      E@      E@     �A@      8@      D@      E@     �C@     �F@      M@      P@     �M@     �G@      N@     �E@      L@      O@     �N@      T@     �K@     @R@      N@      P@      T@     �R@     �T@     �X@     �[@     @V@     �S@     @X@     �W@     �V@      `@      Y@     �\@     @`@     @_@     �Y@     �`@     �`@      c@      `@      b@     �h@     @c@     @b@     �g@      f@     �d@     @h@      e@     `h@      h@     �f@      k@     @k@     �m@      k@     `o@     @m@     �n@     �p@     �p@     �q@     @s@     �q@     �q@      u@     @t@     �s@     pu@     @u@     pv@     �w@     �w@     �w@     @y@     0|@     �|@     �z@     �|@     @~@     ��@     H�@     �}@     �@     ��@     x�@     ��@     ��@     �@     h�@     ��@     ��@     �@     ��@     �@     ��@     ��@     8�@     ��@     0�@     ��@     0�@     $�@     ��@     X�@     ��@     0�@     p�@     ܙ@     4�@     4�@     x�@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ̡@     $�@     8�@     ��@     �@     �@     0�@     ©@     �@     �@     ��@     �@     ߱@     �@     ��@     �@     /�@     ��@     g�@     Q�@     ��@     Ƽ@     B�@    ���@    ���@     ��@     �@    �,�@      �@     ��@     ��@     '�@     &�@     ��@     
�@     7�@     p�@     S�@     ��@     ��@     @�@    �]�@     �@    �i�@    ��@     �@    ���@     n�@     ��@     ޹@     2�@     �@     �@     ��@     ��@     \�@     ,�@     ��@     ��@     ٳ@     �@     G�@     X�@     ͵@     �@     )�@     ��@     T�@     ��@     Ӽ@     ��@     ��@      �@     �u@     t@     �d@      E@      ;@      5@      .@      @      @        
�
predictions*�	   �yƿ   @@�?     ί@!  <�U�S�)`	�#�P-@2�
�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74�ji6�9���.����ڋ��vV�R9��T7��������6�]����T7��?�vV�R9?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�P�1���?3?��|�?�������:�
              �?      �?      @      @      "@      *@      .@      <@      A@     �I@     �P@     �[@     `a@     `j@     �l@     @r@     0q@      p@      p@     �d@     �b@     �b@     �Y@      V@     �W@      T@      Q@     @Q@     @Q@      J@      ?@      F@      ;@      =@      7@      5@      @      *@      "@      *@      (@      $@      "@      @      $@      @      �?      $@      @      @       @      �?      @              @      @      �?      @               @              @      �?      �?               @              �?      �?              �?              �?              �?              �?      �?              @               @      @      �?       @      �?       @       @      �?       @              �?      �?      @      @      @      @      @      @      @      �?       @      @       @      *@      "@      @      @      (@      @      @      0@      (@      0@      0@      *@      5@      .@      0@      (@      0@      3@      .@      1@      1@      3@      2@      4@      (@      4@      4@      ,@      .@      2@      0@      *@      &@      &@      ,@       @       @       @      $@       @      @       @      @      @      @       @      @      @      @       @       @      �?      �?      �?       @              �?        ぷ�1       �[�	��Vm���A*�c

mean squared error��=

	r-squared�HC>
�L
states*�L	   ��O�    ɤ@    c2A!Z���cݶ�)l���@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @     �o@     ��@     ��@     Ѐ@     Ё@     ��@     ��@      �@     ��@     ԙ@     �@     �@     n�@     f�@     ��@     ��@     �@     Բ@     ��@     *�@     9�@     "�@     8�@     A�@     f�@     �@     ��@     *�@     ��@     6�@     ��@     y�@     ��@     L�@     �@     %�@     
�@     ��@     J�@     ��@    �J�@     Q�@     ��@     k�@    ���@    �$�@    ���@    ���@     [�@     Z�@    ��@     �@     ��@     �@     t�@     K�@     �@     (�@     ��@     ��@     :�@     ��@     _�@     @�@     ͷ@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     X�@      �@     h�@     @�@     £@     ��@     �@     8�@     �@     $�@     �@     @�@     �@     |�@     �@     t�@     ܓ@     4�@     �@     p�@     ��@     đ@     ��@     ؉@     p�@     ȏ@     p�@     ��@     ��@     h�@     ��@     X�@     P�@      �@     8�@     ��@     h�@     0@     @@     ��@     �|@     �|@     �}@     �y@     pz@     �w@     0w@     �y@     0w@     �v@     pu@      v@     s@     �q@     �r@     �q@     �q@     �n@     �p@     `p@     @n@      p@     �n@      j@      i@     @j@     @l@      r@      o@     �e@      d@     �f@      g@     �c@     �d@      c@     `a@     �a@     �c@     `b@     �a@      _@      `@     �^@     `a@     �]@     @W@     �[@     �Y@     �X@     @Y@      Y@     �V@     @W@      W@     �Y@     �S@     �Q@     �S@      S@     �Q@     �R@     �Q@      Q@     �R@     �P@     �R@      R@      M@     �L@     �J@     �M@     �F@      M@     �S@     �D@     �G@      J@     �G@      D@      C@     �C@      G@      ;@      D@     �@@      ?@     �D@      C@     �A@      9@      5@      9@      @@     �@@      ;@      <@      8@      7@      7@      1@      1@      8@      >@      <@      8@      1@      2@      0@      8@      1@      ,@       @      .@      *@      .@      2@      2@      4@      2@      5@      $@      0@      1@      &@      &@      2@      4@      (@      $@      (@      @      (@      @      $@      (@      &@       @      &@      ,@       @      *@      @      @      @      @      @      �?      @      @      @      @      @      @      @      @      @       @      @      @      @      (@      @       @      @      @      @       @      @       @       @      @      &@      @      @      @      @      @      @       @       @      @      �?       @      �?      @      @       @     `g@     �l@      @       @      @      @      @      �?      @      @      @      @      $@      @      �?      @       @      @      @       @      @      @       @      @              @      @      (@      &@      @      @      @       @      "@      "@      "@      @      @      $@       @      @      "@      "@       @       @      "@      $@      @      $@      @       @      (@      &@      *@      "@      0@      0@      ,@      ,@      (@      &@      (@      0@      ,@      (@      .@      "@      1@      ,@      .@      2@      0@      *@      $@      .@      7@      2@      <@      1@      ;@      .@      (@      5@      =@      ;@      4@      3@      7@      :@      1@      ?@      A@      5@      :@      <@     �@@      @@      ;@     �@@      A@     �A@      C@     �E@     �K@      D@     �H@     �@@     �@@     �G@     �G@     �J@     �M@     �N@      R@     �P@     �O@      P@     �Q@     @R@     �I@      Q@     �P@     �W@     �P@     �P@      R@     �U@     @W@     �V@      Y@     �V@     @Z@     @[@      `@      _@     `a@      ^@     �Y@      `@     `b@      \@      b@     �d@     �d@     @b@     �d@     `f@      f@     �d@     @g@      f@      i@     �m@     �h@     �l@     @n@     �j@     �k@      p@     �o@     r@     0q@     �q@      s@     �q@     pq@     r@      s@     u@     0v@     �u@      v@     �u@     �w@     x@     �x@     �x@     �z@     `y@      |@     |@      @     �~@     Ѐ@     @�@     �@     �@     ��@     `�@     ��@     �@     ��@     ��@     ��@     ��@     �@     h�@     0�@     ��@     ��@     |�@     \�@     ��@     �@     ��@     ��@     ��@     �@     d�@     �@     ��@     ��@     ��@     ��@     ��@     �@     �@     ��@     ��@     ��@     (�@     ��@     Z�@     t�@     ��@     |�@     \�@     �@     ��@     ��@     ԰@     ��@     ��@     �@     �@     M�@     ��@     [�@     ݸ@     �@     �@     ��@     ��@    ��@    �4�@    �F�@    ���@     ��@     A�@    ���@    ���@     ��@     �@     2�@     ��@     ��@     q�@     ��@     p�@    ���@     ��@    ��@    �a�@    ��@     &�@    �l�@     P�@     ��@     ��@     ��@     �@     ׸@     �@     ��@     �@     �@     ��@     G�@     ��@     ��@     ݲ@     ��@     �@     س@     ��@     ��@     ĵ@     ��@     ;�@     ��@     �@     @�@     Ρ@     �@     x�@     p|@     `v@     `n@     �Y@      D@      7@      1@      $@      @        
�
predictions*�	    ��ÿ   ��7�?     ί@!  `�G_@)���0G@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9��f�ʜ�7
��������Zr[v��I��P=��f�ʜ�7
?>h�'�?��ڋ?�.�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?�������:�              �?      �?      @      @       @      $@      $@      &@      *@      2@      A@      G@     @P@     �R@      Y@     @^@     �[@     �b@     �]@     �U@     @Y@     @T@      P@     �N@      K@      J@     �C@     �@@      ;@      <@      5@      8@      4@      (@      *@      $@      1@       @      &@      $@      $@      $@      @       @      @      @      @      @      �?      @      @      �?       @       @      �?      �?      @       @      @       @      @              �?       @      �?              �?       @              �?              �?               @              �?              �?              �?              �?               @              @       @       @       @       @              @               @              @              @      �?      @      @      �?      @      @      @      @      "@      @      @      @      $@       @      "@      (@      "@      (@      (@      .@      4@      0@      4@      4@      8@      ?@      =@      ?@      A@     �H@     �G@     �G@     �P@      L@     �T@      S@     @Q@      T@      U@      U@     �U@     �T@     �T@     �T@     @Q@     �Q@      N@     �I@     �K@      J@      J@      G@      F@     �B@     �D@     �@@      ?@      =@      =@      4@      :@      0@      .@      @      @      *@      @      @      @      @      �?      @       @      �?       @      �?      �?              �?        x��pB1      ��N	n5xm���A*�b

mean squared error#�=

	r-squaredxX>
�L
states*�L	   @��   �2�@    c2A!�X"F4b��)*��7��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&               @     Pq@     @�@     ��@     ��@     x�@     �@     �@     ��@     �@     `�@     ��@     �@     ��@     �@     ��@     +�@     ձ@     ��@     ��@     ��@     x�@     ?�@     ܱ@     Z�@     F�@     L�@     ��@     3�@     '�@     *�@     ��@     ��@     �@    ���@     i�@     ��@    ��@      �@    �z�@     h�@     ��@    ���@     ��@     �@    �^�@    ��@    ���@    �B�@    �&�@    ���@    ���@    �@�@    �G�@     �@    ��@     w�@     %�@    � �@     ��@     �@     �@     b�@     ��@     ~�@     #�@     ��@     �@     ��@     �@     ��@     >�@     P�@     ��@     n�@     ��@     ̢@     <�@     P�@     �@     �@     ��@     �@     ��@     ��@      �@     �@     Ԕ@     P�@     ��@     `�@     ̒@      �@     8�@     (�@     ��@      �@     Џ@     X�@     h�@     H�@     P�@     ȇ@      �@     ��@     Ѕ@     �@     ؂@     (�@     ��@     x�@     ��@     �@     �~@     0~@     P�@     p}@     �|@     �z@     py@     �y@     �y@     �w@     �w@     �v@     �s@     �u@     `s@     �t@     �r@     �r@     �p@     Pq@     @o@     `o@     �q@     �q@      n@     `n@     �m@      j@     @j@     �i@     �v@     `g@     �i@     �g@      g@     �c@     �f@     `d@     @f@     @d@     �d@     �a@     �a@     �^@     @`@     �Z@     �`@     @`@     �]@      _@      \@     �Z@     �X@      X@     @^@     @U@     �V@      U@     �V@     @W@     �S@      T@     �T@     �R@      O@     �R@     @R@     �P@     @S@      L@     �L@      M@      Q@      J@      P@      I@      E@      M@      E@      @@      M@     �@@     �C@      H@      E@      ?@     �B@     �E@      E@     �A@      >@      A@      ?@     �A@      A@      <@      C@      A@      =@      6@      6@      6@      >@      >@      1@      :@      4@      5@      1@      ;@      6@      5@      .@      3@      4@      0@      5@      *@      (@      ;@      *@      &@      5@      .@      2@      ,@      *@      3@      $@      @      .@      @       @      (@      (@       @      *@      $@      @      @      @      .@      (@      @      (@      $@      "@      "@      @       @      @      @      @      @      @      @      @      @       @      @      @      $@      @       @       @       @      @      @      �?      @      @      @      @      @      @      @       @      @      $@      @      @      @      @       @       @      @      @      �?       @     `g@     �o@       @      @      �?      @      @      @      @      @      @      @       @              @      @      @      @      @      @      @      @      @       @      @       @      @      @      ,@      @      @      "@      @      "@      @      @      @      @       @      $@      @      @      @      .@      &@      &@      $@      @      @      &@      $@      &@      "@      "@      &@      ,@      ,@      @      "@      3@      ,@      $@      *@      3@      $@      2@      @      3@      0@      .@      9@      7@      :@      3@      5@      6@      ,@      3@      5@       @      .@      5@      8@      6@      1@      8@      :@      5@      ;@      8@      =@     �B@     �A@      :@      D@      =@      H@     �A@     �D@     �F@      K@      E@     �F@      G@     �C@     �B@     �I@      I@      I@     �L@     �H@      K@     �L@      P@      K@     @T@     �Q@     �S@     �Q@     @R@     �S@     �X@      W@     @W@     �W@     @V@     �S@     �[@     �V@     @\@     �X@      Y@     @]@     @_@     @\@      b@     �^@     @a@     �b@     @_@     `b@     `d@     �c@     `b@      e@     �c@      g@     �g@      f@      g@     �f@     �h@      j@     �f@      o@      l@     @m@     `m@      m@     �n@     @o@      n@     pp@     pq@     �p@     Pq@     �r@     �s@      t@     �t@     �t@     �w@     @u@     �w@     �x@     �z@     0z@     |@      |@     �~@     �~@     �}@     �@     �@     `�@     h�@     ��@     H�@      �@     P�@     X�@     Ѕ@     ��@     ��@     P�@     8�@     8�@     Њ@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ��@     �@     |�@     ��@     Ȓ@     ��@     L�@     Е@      �@     h�@     0�@     ��@     \�@     `�@     ��@     D�@     Ԡ@     ,�@     .�@     2�@     ��@     T�@     ޤ@     V�@     �@     ��@     R�@     `�@     Ҭ@     :�@     	�@     )�@     ��@     D�@     ��@     [�@     ��@     ��@     �@     @�@     k�@     ��@    �P�@     `�@     ��@    ��@    �X�@    �E�@    �[�@    �u�@     ��@     \�@    ���@     ��@    ���@    ���@    ���@    �_�@    �h�@    �f�@    �;�@     �@    ���@    ��@    �4�@     ƽ@     ֺ@     ޶@     e�@     z�@     �@     p�@     g�@     |�@     ֱ@     �@     �@     ��@     !�@     ��@     �@     ��@     ��@     d�@     �@     �@     ��@     ��@     ��@     X�@      ~@     �v@      o@     �Z@     �B@      7@      6@      3@      @        
�
predictions*�	   ��̿    ^n@     ί@!  'Y�)��M8@2�
�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.��O�ʗ��>>�?�s��>U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��%�V6?uܬ�@8?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?��tM@w`<f@�������:�
               @       @      @      �?      @      .@      3@      3@      >@     �I@      I@      K@     @U@      [@     �X@      a@     @g@     `o@     q@     @q@     �p@     `l@      h@     `i@      d@     �_@      W@      T@     �R@     �P@      J@      I@      4@     �@@      9@      A@      ,@      2@      ,@      (@      *@      (@      (@       @      �?      @       @      @       @      @       @      @       @      �?      @      @      @      @      �?      �?      �?       @              �?              �?       @              �?              �?      �?      �?              �?              �?              �?       @              �?              �?              �?              @      �?       @      �?      �?              @      @              @      @      @      @      $@       @      @      @       @      "@      "@      $@      @      &@      @      (@      *@      &@      &@      $@      &@      "@      ,@      3@       @      $@      ,@      1@      "@      (@      6@      4@      @      &@      *@      "@      &@      $@      $@      (@      *@      @      @      @      ,@      &@      @      @      @      @      @      @      �?      @      @      @              �?       @              �?      @              �?      �?      �?      �?       @              �?        �b3      ���+	.Ɠm���A*�e

mean squared error�=

	r-squared�ps>
�L
states*�L	   `C8�   `1�@    c2A!n�b9�(��))�ԺO�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �?     `h@     H�@     ��@     t�@     �@     ��@     8�@     `�@     А@     ̔@     ��@     ��@     �@     Y�@     Y�@     v�@     �@     7�@     �@     ��@     ��@     �@     ѱ@     G�@     ܱ@     ��@     ��@     ��@     ��@     ��@     6�@     ��@     �@     ��@    ���@    ��@    �z�@     ��@     ��@    �J�@     F�@     ��@     J�@    ���@    ���@     )�@    ���@     ��@     ��@     5�@     �@    �%�@    �#�@    �7�@     ��@    ���@     ��@    ��@     ��@    �(�@    ���@     þ@     P�@     y�@     ��@     ɷ@     ޵@     j�@     �@     ��@     ��@     R�@     ��@     ��@     ��@     �@     j�@     ��@     ��@     �@     ��@     ��@     ��@     `�@     X�@     l�@     D�@     ��@     ��@     ��@     0�@     ��@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ȍ@     ��@     ��@     ��@      �@     �@     @�@     ȅ@     ��@     `�@     �@     `�@      �@     (�@     Ѐ@     ��@     p�@     8�@     �~@     �}@     p}@     �{@     �z@     @y@     0v@     �y@     �t@     �v@      x@      u@     �u@     ps@     �s@     �r@     pr@     Pr@     pq@     �p@     �r@     �q@      o@      n@     �m@     @l@      l@      w@      j@     �j@      f@     `h@     �g@     `h@     @g@      c@     @d@     �c@     `c@     @c@     `e@     @d@     @a@      _@     @^@      b@      ^@     �^@     �Y@     �[@     �Y@     �\@     �\@      ]@     @Y@     @V@     �S@      U@     �[@     �U@     �X@     �V@     @T@     �T@     �P@     �P@     �P@     �N@      Q@     �M@     �K@     �I@      G@     �L@      N@      I@     �L@     �E@     �E@      K@     �H@      B@      I@     �D@     �C@      C@      @@      ?@      >@      C@      >@     �G@      <@      C@     �@@      8@     �C@      2@      5@      ;@      7@     �@@      1@      :@      5@      =@      9@      6@      1@      <@      :@      6@      1@      0@      &@      5@      3@      3@      5@      1@      4@      1@      0@      6@      *@      ,@      1@      ,@      2@      *@      (@      $@       @      (@      $@      *@      0@      &@       @      *@      &@      @      @      @      @       @      @      "@      $@       @      "@      "@      @      @      @      @      @      $@      @      "@      @      @      "@      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      @      @      @      @       @      @     �l@     Pq@      @      "@      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      @       @       @       @      @      @       @      @      �?       @      @      @       @      $@      "@      "@       @      &@      (@       @      ,@      @      "@      $@      @      @       @      &@      (@      (@      $@      $@       @      ,@      "@       @      &@      0@      4@      $@      0@      4@      (@      "@      ,@      ,@      2@      2@      .@       @      1@      0@      $@      7@      3@      4@      ?@      9@      4@      4@      =@      9@      6@      A@      .@      :@     �A@      >@      B@      F@      @@      E@      F@      E@     �C@      I@      H@     �D@      G@      M@     �I@     �I@      K@      L@     �K@     �L@     @Q@     �I@     �P@      S@     �Q@     �P@     @U@      N@     @P@      O@     �U@     �P@      V@     �U@     �U@     �V@      X@     @W@     �V@     �V@     �Z@     @]@      ]@     �Y@     �`@      ^@     �`@      `@      a@      _@     �b@     �`@     @d@     `a@      a@     @e@     �d@      e@     �f@     `g@     �h@     `k@     @f@     �j@     `k@      l@     �m@     �o@      o@      n@     �k@     �k@     �p@      r@     �r@     @o@     �p@      q@     Pt@     �t@     �s@     @t@     �u@     �v@     `x@     �v@     �x@     �x@     P{@     �z@     P{@     �{@     `}@     �~@     ��@     ��@     ��@     �@     Ђ@     ��@     �@      �@     ��@     ؇@     ��@      �@     ��@     ��@     ��@     ��@     P�@     H�@     �@     8�@      �@     ��@     ��@      �@     ��@     ��@     L�@     l�@     ��@     L�@     ��@     `�@     ��@     ,�@     ��@     ��@     t�@     V�@     ��@     ؠ@     ��@     �@     ��@     ��@     �@     \�@     ��@     ��@     4�@     ��@     #�@     ��@     �@     !�@     ��@     ��@     ݶ@     ��@     F�@     ��@     �@     �@     ��@    ���@     ��@     ��@    ���@    ���@     P�@     {�@     :�@     L�@    �%�@     �@     ��@      �@     ��@    �`�@    ���@    �A�@     ��@    ���@     &�@    �Y�@     ��@     ��@    ���@    �4�@     ½@     �@     ��@     ��@     -�@     ��@     ��@     �@     Ų@     ��@     %�@     >�@     j�@     s�@     �@     =�@     /�@     �@     ŵ@     L�@     ��@     ׷@     ��@    ���@     �@     ��@     ��@     �u@     0s@     �j@      X@     �A@      :@      .@      .@      @        
�
predictions*�	   �hbҿ   �,@     ί@!  n���a@)��%{�O@2�_&A�o�ҿ���ѿ�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9��x?�x��>h�'���ߊ4F��h���`FF�G ?��[�?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�6v��@h�5�@�������:�              �?               @      @      @      @      @      @      $@      "@      &@      3@      0@      7@     �A@      @@     �D@     �@@      H@     �C@      M@      J@     �I@     �G@      D@     �H@     �H@     �A@      M@     �C@      C@      G@     �D@      J@      ?@      >@      :@      >@      6@      0@      6@      1@      5@      &@      .@      1@      0@      @      @      0@      (@      *@       @      @      ,@       @      "@      @      @      �?      @      �?      @      @              @               @      @              �?      �?              �?      �?      �?       @              �?              �?              �?              �?              �?              �?      �?      �?              �?       @              �?               @       @              �?       @      �?      @              @      �?       @       @      @       @      @      �?       @      @      @      @      @      @      &@      @      @      .@      "@      "@      &@      .@      7@      *@      5@      1@      7@     �C@      ?@      5@     �@@     �G@     �I@     �G@      K@     �P@      M@     �T@     �R@     �V@     �W@     @T@     �S@      Y@     �Y@     @^@     �\@      Z@     �U@     @X@     @S@     �Q@      P@     �E@     �H@      G@     �F@     �E@      B@      @@      A@      4@      9@      8@      :@      5@      *@      ,@      $@      *@      *@      @      @      @      @       @      @      @      �?      @              �?       @      �?      �?       @      �?              �?        ��'Ʋ3      _  	9Юm���A*�g

mean squared error�=

	r-squaredt�j>
�L
states*�L	   �
y�    ��@    c2A!ֲ�6`���)��g_��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @     `x@     �@     ��@     H�@     8�@     @      �@     H�@     ��@      �@     �@     p�@     3�@     �@     ��@     f�@     ��@     ��@     ��@     ��@     ��@     ��@     ذ@     A�@     Ұ@     ��@     Ű@     E�@     ��@     B�@     �@     &�@     ��@    �2�@    �6�@     ��@     ��@     ��@     ��@    ���@     ��@     $�@     ��@    �{�@    �X�@     p�@     �@     �@     -�@    ���@    ���@    �W�@    ���@    ���@     t�@     ��@     p�@    ���@     �@      �@     ��@     !�@     ��@     '�@     A�@     9�@     M�@     Գ@     c�@     ��@     �@     ʫ@     �@     n�@     ~�@     ԥ@     >�@     �@     Ρ@     d�@     &�@     <�@     ��@     t�@     ��@     <�@     ��@     ̕@     �@     ̓@     ��@     (�@     4�@     ,�@     ��@     ��@     X�@      �@      �@     8�@     (�@     @�@     P�@     ؇@     ��@     0�@     �@     Ȅ@     X�@     Ȅ@     ��@      �@     ��@     ��@      �@     �@     P�@     �}@     �@     �|@     �z@     0z@     �{@     `y@     �{@     0x@     �x@     Py@     �y@     0v@      w@      u@     �t@     �t@     �r@     �r@     `r@     �r@     �o@     @p@     �p@     �p@     `l@      n@     �j@     �t@     �r@     �i@      h@     �h@      g@     �g@     @g@      e@     `d@      e@     `b@     �f@     `c@     �e@     �c@      b@     �^@      ^@     �]@     �^@     @]@     @]@     �Z@      `@     �X@     �Y@     �Z@     �Y@     @Y@     �Y@     @W@     @X@     �T@     �S@      V@     @U@     �R@      P@     �Q@      M@     �Q@      P@     �T@      J@     �Q@      L@     �G@      I@      M@     �K@     �F@      C@     �B@     �J@     �D@     �L@      H@      F@     �E@     �A@      K@      <@      D@      ?@      D@     �B@      E@      :@      ;@      >@      8@      B@     �C@      =@      5@      >@      ;@      0@      0@      ,@      ,@      4@      .@      9@      1@      4@      8@      ?@      3@      5@      0@      1@      ,@      6@      2@      *@      8@      2@      &@      @      *@      0@      .@      &@      "@      (@      *@      ,@      @      .@      &@      &@      &@      &@      @      @      *@      &@      @      &@       @      $@       @      @      *@      @      @      @      @      @      @      $@      "@       @      @      @      @       @       @      @      @      @      @      @      @      "@      @      $@      @      @      @      @      @       @      @     @l@     �s@      @      @      @      @      @      @       @      @      @      @      �?      @      @       @      $@       @      @      @       @      @       @      @      @      @      $@      @      "@       @      @      "@      @      @      *@      $@       @      ,@      "@       @      @      @      &@       @      *@      @      "@      2@      &@      1@      "@      0@      (@      "@      .@      (@      3@      0@      ,@      *@      7@      *@      .@      8@      0@      2@      0@      <@      1@      1@      =@      5@      0@      :@      .@      7@      7@      7@      ;@     �A@     �@@      <@      7@      B@     �@@     �B@      F@      :@      ;@     �E@     �@@     �C@     �D@     �A@     �J@      F@      F@      L@     �E@     �H@     �P@      M@      O@      R@     �K@     �S@     �O@     �P@      O@      M@     @Q@     �I@      M@     @R@     �R@     �V@     �X@     �U@     @U@      V@      V@     �Y@     �[@     �X@     �W@     @\@     �]@      \@     �Y@     �\@     @]@     �[@     �d@     �`@     `b@      ^@     �a@     @e@     @d@     �a@     `a@     `b@     �f@     �e@      f@      k@      i@     �i@     `e@     `j@     `k@     `l@     �i@     �l@     �l@     @k@     �o@     @p@     q@     �q@     �q@     @r@     s@     �r@     �u@     �t@     0s@     `u@     Pu@     @w@      v@     0y@     p{@     �x@      {@     �y@      {@     |@     0~@     �~@      }@     �@      �@     ��@     (�@     ��@     ��@     І@     �@     x�@     �@     X�@     �@     �@     ؇@     P�@     ��@     P�@     ��@     Đ@     H�@     ��@     L�@     L�@     ��@     ��@     ĕ@     8�@     �@     ��@     L�@     ��@     X�@     p�@     Ԛ@     �@     `�@     ��@     ��@     ̞@     �@     P�@     J�@     b�@     �@     F�@     r�@     >�@     ި@     L�@     ��@     `�@     
�@     ��@     ��@     %�@     �@     ̵@      �@     [�@     ��@     ֺ@     u�@     q�@    ��@     #�@     n�@     ��@     :�@     ��@     N�@    �w�@     ��@    �g�@    �>�@     ��@    ���@     �@     ��@    �I�@     ��@    ���@     ��@    ���@     @�@     n�@     C�@    ���@      �@     x�@     ��@     �@     [�@     �@     _�@     K�@     �@     �@     �@     ��@     ��@     [�@     ˲@     ��@     J�@     ;�@     ��@     ��@     �@     1�@     �@    ���@     >�@     �@     �~@     �u@     @r@      k@     �Z@     �H@      5@      $@      2@      @        
�
predictions*�	   ���׿   @@I@     ί@!  �o�V�)�h�b'{:@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��T7����5�i}1�6�]���1��a˲���Zr[v��I��P=���_�T�l׾��>M|Kվ5�"�g��>G&�$�>8K�ߝ�>�h���`�>6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�6v��@h�5�@�������:�              �?      @      @       @      @      @      @       @      @      5@      =@      ;@      D@     �G@     �K@     @P@     �U@     @T@     @W@     �Z@      b@     �[@     @_@     �_@     �`@     �`@     �a@     �_@      ^@     �a@     @^@     @Z@     �X@     @W@     �S@     @R@     �L@      O@      I@      M@     �B@      <@     �C@      <@      6@      3@      :@      2@      (@      0@      3@      @      "@      @      @      @       @      @      @      @      @      @      @      @      @      �?      �?      �?       @      @      @      @              @              @       @      �?              �?      �?      �?              �?              �?              �?               @              �?              �?              �?              �?              �?               @              �?               @              �?      �?      �?      @      �?      @              �?              �?              @       @      �?      @              �?      @      @      @      @      �?      @      �?      @      @       @      $@      "@      @      $@      0@      (@      ,@      &@       @      ,@      *@      .@      &@      0@      9@      *@      2@      7@      9@      7@      7@      <@      @@      9@      5@      9@      4@      7@      7@      ?@      1@      6@      0@      7@      1@      8@      *@      &@      ,@      $@      &@      *@      ,@      1@      @       @       @      @      @      @      @      @      @      @      @      �?      @      @      �?               @               @              �?              �?              �?              �?        �a3      ���+	��m���A*�e

mean squared error��=

	r-squaredT�>
�L
states*�L	   ���   ���@    c2A!:�c���)M�ڵo�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              D@     ��@     �@     �@     p�@      �@     �@     (�@     ��@     ��@     ��@     ʣ@     S�@     �@     '�@     �@     y�@     ��@     ��@     ��@     ^�@     ��@     .�@     �@     l�@     ��@     ��@     
�@     ��@     �@     %�@     ��@     ��@     Ƹ@     �@    �7�@    ���@    �>�@     �@     {�@    ��@     ��@     ~�@    ���@     ��@    ���@    ���@    �5�@    ���@     �@     ��@    �s�@     ��@     =�@    �=�@     ^�@     o�@    ���@     a�@     ��@    �%�@     ��@    �v�@     2�@     ?�@     ��@      �@     �@     ɴ@     5�@     �@     �@     ֯@     ��@     6�@     "�@     j�@     ��@     4�@     ʣ@     ΢@     ��@     P�@      �@     ��@     ܝ@     `�@     $�@     ܛ@     ��@     �@     ĕ@     l�@     ��@     ܒ@     |�@      �@     @�@     ��@     X�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     �@     `�@     ��@      �@     �@     ؅@     ��@     Ȃ@     H�@     8�@     ��@     �~@     �}@     ~@     �|@     �{@     �}@     `{@     �{@     �|@     �x@     @x@     px@     �y@     �u@     pw@     0v@     �v@     �t@      t@     Pu@     @s@     �r@     �q@     pq@     �p@     `q@     �o@      l@     `k@      v@     �p@     �l@     `m@      i@     �g@     �g@     `g@      f@     `f@     �e@     �g@      g@      b@      c@     @a@      d@     �a@      c@     `b@     �`@      ]@     �_@     @[@      [@     �Z@     @]@     @\@     �\@      W@     �Y@     �W@      \@     �Y@     �T@     �V@     @Y@     �V@      O@     @T@      U@      P@      T@     �Q@     @Q@     �P@     �P@     �O@     �O@      L@      G@     �N@      G@     �J@     �L@      J@      I@      I@     �D@      B@     �G@      D@     �I@      :@     �B@      E@     �G@     �A@      >@      =@     �B@      ?@      :@     �A@      1@      A@      ;@     �D@      6@      9@      *@      ;@      ?@      &@      <@      :@      :@      6@      (@      2@      6@      0@      ;@      2@      1@      &@      0@      ,@      5@      ,@      *@      @      "@      $@      1@      &@      $@      $@      ,@      &@      2@      .@      2@      "@      $@      ,@      @      &@      $@      @      "@      "@      @      $@       @      @      @      (@      @      @      "@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      @       @      @       @       @      @      @      @       @      @     `p@      u@      @       @      @      $@      "@      �?      @      @      @       @      *@      @       @       @      @      @      $@      @      $@      @      @      @      @      &@       @      @      &@      @      @      @      @       @      @       @      "@      ,@      @      ,@      @      &@      @      @      *@      $@      (@      $@      0@      1@      "@      .@      ,@       @      2@      .@      .@      0@      1@      .@      8@      8@      3@      4@      ,@      ;@      1@      8@      &@      0@      :@      :@      :@     �@@      ?@      7@      :@     �A@      =@     �@@     �C@      D@      :@     �C@     �F@     �C@      G@     �A@      H@     �N@      B@     �D@      D@      H@     �B@     �F@     @P@      J@     �K@     �L@     �I@     �F@      Q@     �Q@     �J@     @R@     �S@     �P@     �R@      U@     �R@     @T@     �R@     �R@      T@      U@     �V@     �Q@      U@     @W@     �Z@     @Y@     �W@     �\@      X@      ^@      `@     �`@     �^@     @_@     �`@      ^@     �`@     `a@      `@      c@     @b@     @e@     `f@      f@     �f@      f@      f@     �j@     `h@     �j@     @h@     @i@      h@     �j@      o@     �j@     �o@     �p@      m@     @p@     �p@     �p@     Pq@     q@     �r@     ps@     �s@     t@     �u@      r@     Pw@     �u@     0x@     �x@     �x@     p@     �|@     �{@     `}@     �@     �|@     @@     �}@     �~@     ��@     H�@     (�@     �@     ��@     x�@     ��@     H�@     ��@     ��@     ��@     �@     X�@     H�@     Ē@     ��@      �@     P�@     ��@     P�@     ��@     ��@     Џ@     �@     �@     |�@     Ȓ@     0�@     ��@     `�@     ��@     ��@     \�@     (�@     �@     d�@     P�@     <�@     0�@     Ԣ@     Ԣ@     ��@     �@     ܥ@     .�@     (�@     ��@     ��@     ��@      �@     ��@     ��@     8�@     �@     R�@     B�@     Ӵ@     [�@     X�@     S�@     �@     ˾@    ���@     �@     h�@    ���@     ��@    ���@     ��@    ���@     H�@     J�@     m�@     j�@    �	�@     t�@    �'�@     E�@     I�@     ��@     ��@     �@     ��@    ���@     Z�@     ��@    ��@     D�@     ��@     2�@     ��@     a�@     Ե@     �@     6�@     ��@     �@     \�@     Q�@     ��@     �@      �@     ߲@     ��@     ��@     ��@     �@     ӷ@     ~�@     �@     G�@    ���@     0�@      �@     �~@     �v@     �p@     `j@     @[@     �L@      5@      4@      0@      @        
�
predictions*�	   `��տ   �I
@     ί@!  P�U�Q@)Ȼ�O,O@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��S�F !�ji6�9��1��a˲���[��1��a˲?6�]��?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@�Š)U	@u�rʭ�@�������:�              �?               @       @      @       @      �?      @       @      @      "@      @      &@      ,@      6@      1@      8@      6@      F@      H@      F@     �N@     �T@     �V@     @[@     �a@     @\@      b@     �[@      [@     �X@     �X@      Z@     �U@     �W@      R@      K@     �K@     �N@      D@     �@@      B@      =@      @@      8@      :@      1@      (@      (@      2@      *@      @      &@      "@      *@       @       @      @      @      @      "@      @      @       @      �?       @       @      @      @      �?      �?              �?      �?       @              �?               @              �?              �?              �?              �?      �?      �?              �?      �?               @      �?              �?              �?               @      �?       @      �?       @      �?      �?      @       @       @      @      @      @       @      @      @       @       @      @      @      ,@      @      $@      6@      $@      .@      *@      .@      &@      &@      .@      ;@      8@      8@      @@      6@      @@      @@      A@     �B@      I@      F@      G@      J@      F@      K@     �L@     �C@     �K@     �D@     �F@     �A@      H@     �G@     �F@     �C@      B@      E@      7@      B@      ?@      >@      <@      =@      >@      6@      4@      1@      0@      .@      &@      0@      $@      ,@      @      @      @      @       @      @      @      �?      @      �?      @      �?      �?      �?      �?      �?              �?              �?        �@ا�2      ����	jh�m���A*�e

mean squared errorc=

	r-squared �>
�L
states*�L	   ����   @�@    c2A!����õ��).�6��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             @R@     L�@     4�@     d�@      �@     X�@     ؀@     ��@     ��@     ��@     ��@     ��@     ��@     +�@     )�@     �@     I�@     1�@     ��@     Y�@     }�@     ��@     ��@     5�@     ��@     ��@     ��@     7�@     �@     _�@     ��@     ��@     ��@     ܸ@     �@     ��@    ��@     ��@     �@     �@    ���@    �U�@    �L�@    �;�@    ���@    ���@    �\�@    ���@     '�@     ��@     G�@    �a�@    ���@     ��@    ��@     /�@    ���@    ���@     V�@    �n�@     _�@     D�@     )�@     %�@     ޻@     ��@     �@     ��@     ��@     	�@     Z�@     =�@     �@     ^�@     ީ@     l�@     ��@     b�@     ��@     .�@     Ԣ@     ��@     ��@     �@     ��@     D�@     �@     d�@     ��@     ��@     ��@     l�@     ��@     ��@     ��@     P�@     ��@     ܒ@     l�@      �@     ��@     А@     ��@     h�@     ȉ@     ��@     ��@     ��@     �@     �@     ��@     ��@     ȃ@     ؃@     H�@     ��@     0�@     ��@     P�@     ��@      ~@     �}@     �}@     ��@     �@     p}@     �|@     P{@     z@     Pw@      {@     �v@     Pw@     �w@     `v@     �s@      u@     �s@     ps@     �s@     Ps@     �r@     �q@     Pq@      q@     �p@     @p@     �p@     �t@     Pr@      l@     �l@     `k@      h@     �i@     �g@     �f@     �f@      h@     `g@      e@     `f@     �d@     @e@     �b@      c@     �c@      b@     ``@      \@     @a@     �c@     �]@     �^@     @^@      `@     �X@      \@      [@     @\@     �[@     @W@     �T@      W@     @W@     �T@      R@     �T@      R@      V@     @R@     @R@      V@      S@      J@     �N@     @P@     �Q@     �P@      K@     �K@     �F@      M@     �K@      G@      I@      J@      B@     �@@      H@     �L@     �D@      F@      J@      I@     �D@     �C@     �B@     �@@      :@      B@     �A@      =@     �A@      :@      6@      7@      5@      >@      3@      >@      4@      9@      4@      7@      7@      7@      @@      4@      (@      5@      5@      0@      0@      7@      3@      5@      .@      6@      *@      2@      (@      .@      *@      2@      .@      ,@      2@      "@      $@      $@      (@      $@      &@      .@      @      "@      (@      @      (@      $@       @      (@      (@      @      @      $@      "@       @      @      @      "@      @      @      @      @      "@      @      @      @      @      @      @      @       @      @      @      @      @      @       @      @     q@     @t@      "@       @       @      &@      @      @      @      @      @      @      @       @      @       @      @       @      @      @       @       @      @      @      "@       @      @      @      @      *@       @      (@      "@       @      @      $@      $@       @      *@      &@       @      @      0@      *@      "@      1@      (@      ,@      0@      "@      &@      *@      2@      1@      *@      .@      2@      .@      8@      &@      <@      9@      4@      1@      .@      ,@      7@      9@      9@      7@      =@      ;@      =@     �@@      A@      A@      A@      ;@      D@      ;@      :@      A@     �B@      A@     �D@      A@      E@     �D@      E@      D@     �J@     �H@      I@      F@      H@     �J@     �M@     �L@     @P@      M@     �I@     @R@     �M@      O@     �Q@     �R@     �Q@     @R@     �S@      S@      Y@      R@     �W@      U@     �P@      V@     @X@     �W@     �]@     �Y@     �V@      [@     �^@     �[@     @^@     �_@     �^@     @`@     @a@     @`@     �`@     `a@     `f@     �c@     `c@      g@      c@     @c@     @h@     �i@     �e@     �j@     �h@     �i@     �e@     `h@      k@     `j@      j@     �h@     �o@     �m@     �o@      o@      p@     �q@     �p@     �p@      s@     �q@     0s@     �u@     �t@     �x@     @y@     �x@     `y@      y@     pw@     �{@     �x@     pz@      z@     �{@     �|@     0}@     �@     �@     �@     ��@     @�@     (�@     �@     ��@     �@     ��@     �@     Ȉ@     `�@     �@     8�@     ��@     Ȇ@     ��@     0�@     8�@     Ћ@     Ȋ@     P�@     @�@     H�@     ��@     ԑ@     ��@     ��@     ܒ@     p�@     ē@     D�@     $�@     ̗@     ��@     l�@     |�@     |�@     ��@     ��@     ��@     f�@     ��@     Π@     \�@     *�@     �@     Ҥ@     8�@     ��@     ��@     �@     �@     �@      �@     ��@     ��@     ͳ@     ׳@     �@     �@      �@     ��@     ͹@     �@     &�@     ��@    ���@     ��@    ���@     ��@     �@    �I�@     2�@     Q�@    ��@     u�@    ���@     ��@    ���@    �<�@     f�@     ��@     a�@     ��@     ��@     ��@    ���@    ���@    ���@     W�@     к@     �@     c�@     Ƶ@     ��@     	�@     ز@     ��@     [�@     b�@     ��@     ӱ@     4�@     ñ@     �@     7�@     9�@     ��@     G�@     
�@     u�@     �@     ��@     ��@     n�@     p�@     p�@     �v@     pq@      g@     �T@      J@      0@      (@      $@      @        
�
predictions*�	   @)]ڿ    f;@     ί@!  P�l&P�)w���u�C@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�ji6�9���.����ڋ�f�ʜ�7
�������ߊ4F��>})�l a�>�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�Š)U	@u�rʭ�@�������:�               @              �?              @      @      @      @      @      "@      *@      4@      ;@      D@      D@      >@     �G@     �B@     @Q@     �P@     �W@     @T@     �Z@     �`@     �a@     �h@     @e@     @c@      h@     `e@     �e@     �`@      ]@     �Y@     @X@      R@      P@     �J@     �P@     �G@     �F@      B@     �F@      =@      7@      8@      :@      0@      (@      3@      ,@      2@      3@      ,@      $@      (@       @      $@      @      @      @       @       @      �?      �?      @      @      �?       @      @      @              �?      �?              �?              �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      @      @              �?              �?      @      @      �?      @      @      @       @      @               @       @      @      @      @       @      "@       @      @       @      &@      *@      5@      .@      &@      0@      3@      .@      *@      1@      2@      *@      *@      5@      1@      4@      7@      7@      3@      >@      1@      5@      3@      2@      >@      @@      7@      9@      (@      5@      *@      0@      @      3@      0@      @      @      $@      @      @      @      @      @      @      @      @       @       @       @      @              @      @      �?      @               @      �?      �?              �?      �?               @              �?        ��>�3      �QIH	��n���A*�f

mean squared error�=

	r-squared��>
�L
states*�L	   ����   @"�@    c2A!��t�����)̹��=�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              [@     �@     В@     X�@     ��@     ��@     p�@     ��@     ��@     ,�@     D�@     ��@     8�@     
�@     g�@     ��@     ��@     X�@     �@     �@     �@     ��@     �@     t�@     D�@     ��@     ��@     �@     �@     #�@     d�@     ��@     `�@     ��@     ��@     ��@     ��@     ��@    �~�@    �?�@    ���@     �@     ��@    �E�@     ��@    ��@     ��@    �^�@     ��@    ���@     T�@     ��@    ���@     ��@    ���@    �Z�@     ��@    ���@    �]�@     �@    ���@     ��@    �@�@     ��@     ��@     ٻ@     ߸@     ζ@     ͵@     0�@     2�@     ��@     �@     ��@     ��@     �@      �@     ��@     L�@     ��@     :�@     (�@     ��@     �@     ��@     ��@     D�@     �@     �@     �@     Ș@     d�@     (�@     ��@     ��@     ��@     P�@     d�@     ��@     D�@     h�@     ��@     ��@     P�@     ��@     ��@     ��@     8�@     ��@     ��@     ��@     ��@     �@     ��@     0�@     ��@     ��@      �@     8�@     ȁ@     ��@     @     �~@     �@     |@     0|@     P{@     �y@     0y@     @y@     �x@     y@     �x@     �@     pw@      v@     �s@      t@     �s@     �t@     �s@     �s@     Pr@     `q@     Pq@     �q@     `p@     �n@      n@     `x@     `m@     �m@     Pp@     �j@     `l@     �i@      k@     �h@     �g@     �g@     �i@     �d@     @e@     �g@     �d@     �c@     �`@     �e@      ^@     @a@     �a@      b@     �_@     �a@     �_@     �]@     @_@     �\@     @\@     @\@     �W@     @W@     �X@      Z@     @X@     @W@     @W@     �X@     �Q@     �V@     @W@     �S@      T@      R@     @R@     @P@      N@     �N@     �N@     @R@     �P@     �R@     �K@     �P@      J@     �D@     �H@      G@      N@     �G@      G@      G@      O@      F@     �F@      ?@     �C@      K@     �@@     �A@      A@     �@@      ?@      <@      @@      9@      @@      @@      7@      5@     �E@      6@      8@      4@      1@      4@      A@      5@      6@      3@      7@      7@      9@      0@      1@      2@      .@      1@      3@      0@      ,@      *@      .@      .@      2@      .@       @      &@      .@      &@      "@      .@      *@      1@      .@      &@      .@      (@      @      4@      "@      @      @      @      .@      "@      @      @       @      @       @       @      ,@      @      $@      (@      @      &@       @      "@       @      @       @      @       @      @      @      @      @      @      @      @      s@     pv@      @      @      @      @      @       @       @      @      &@       @      @      @      @      @      @      $@      @      @      @      @       @      @      "@      *@       @      &@      @      @      @       @      *@      @      ,@       @      (@      &@      @      (@      (@      1@      "@      $@      2@      (@      $@      1@      "@      &@      2@      0@      4@      8@      3@      4@      *@      8@      <@      9@      9@      3@      1@      2@      5@      8@      8@      7@      2@      8@      <@      7@      ?@      @@      ;@      >@     �A@     �D@      A@     �B@     �B@     �A@      D@      F@      @@     �F@     �B@     �E@      F@     �G@     �G@      H@     �O@      E@      L@      P@     �K@     @P@      M@      L@     �S@     �P@      N@     @P@     �P@     �K@     �R@     �S@     @R@     �S@     �S@     �U@     �S@      W@     @Z@     �W@     @X@     �[@     �X@     �U@     �\@     @\@     �]@      Z@      `@     @^@     �`@     �`@     �a@     �b@     @c@     `b@     �c@     �c@     @d@     `e@     �e@     �g@     �d@      j@     `h@     �g@      i@      k@     @k@      h@     �n@     �m@     �o@     �l@     pp@     Pq@      r@      r@      s@     0t@     �t@     �r@     �r@     �q@     �t@      u@     �u@      u@     �x@     Px@     `x@     0x@     �y@     `{@     �x@     �@     �{@     ~@     `|@     �|@     �~@     P@     8�@     X�@     (�@     �@     8�@     0�@     ��@     ؄@     ��@     ��@     P�@     X�@     ��@     8�@     �@     p�@     �@     ��@     ��@     �@     p�@     �@      �@     d�@     l�@     L�@     ��@     ��@     ؓ@     �@     T�@     �@     `�@     `�@     @�@     Ԛ@     ��@     �@     �@     P�@     v�@     �@     ��@     L�@     £@     ,�@     �@     N�@     ��@     �@     �@     �@     �@     ��@     �@     G�@     y�@     j�@     �@     �@     �@     U�@     ��@     ��@    �S�@    ��@     ��@     ��@    �J�@    ���@    ���@    �B�@     ��@    �9�@    � �@     u�@     y�@     ��@     ��@    ���@    �e�@     :�@    �J�@    �Y�@    �E�@    ���@    �f�@     N�@     ��@     '�@     ��@     	�@     )�@     
�@     ��@     ;�@     �@     t�@     3�@     ر@     T�@     ��@     �@     �@     .�@     �@     ��@     ˴@     ��@     Ʒ@     d�@     �@     �@    �T�@     H�@     ��@     �@     pu@     @k@     ``@     �P@      B@      @      (@      "@      @        
�
predictions*�	   ��,Կ   �v�
@     ί@!   ��'@)�2H!��E@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ�x?�x��>h�'�������6�]���O�ʗ�����Zr[v���iD*L��>E��a�W�>�ѩ�-�>���%�>})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?�Š)U	@u�rʭ�@�������:�              @      �?      �?      �?      @      @      @       @      "@      2@      *@      7@      >@      @@      D@      C@      H@     �R@      T@     �T@      X@     �Z@     @^@      a@      `@     ``@     @_@     �Y@      U@     �T@     �Q@      P@     @Q@     �P@      N@      J@      B@     �@@     �C@      <@     �B@      9@      6@      9@      5@      1@      1@      (@      0@      $@      (@       @      @      @      "@      @      "@       @      @      @      @      @      @      @       @       @      �?               @      �?       @              �?      �?              �?       @      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?      �?               @              @              �?              �?      @      �?      @      �?      @       @      @      �?      @       @      @      @      @      "@      @      @      @      @      @      &@       @      &@      @      ,@      1@      (@      "@      (@      7@      5@      1@      6@      9@      ;@      ?@     �D@      C@     �F@     �D@     �L@     �F@     �K@     �D@     �F@     �H@     �L@      J@      F@     �N@     �J@      H@     �F@     �E@      C@     �C@      :@     �@@      C@      :@      4@      0@      4@      ,@      0@      ,@      (@       @      (@      $@      @      @      "@      @       @      @      @      @              @      �?       @      @      �?       @              �?      �?              �?              �?        4���23      ��	�F n���A*�f

mean squared error9�<

	r-squared�w�>
�L
states*�L	   @���   �E�@    c2A!B��ӹ���)��d� H�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              r@     ؓ@     X�@     0�@     (�@     ��@     ��@     ��@     <�@     ��@     @�@     ��@     ?�@     �@     �@     ��@     p�@     ��@     	�@     հ@     ��@     Y�@     ��@     ��@     X�@     �@     ��@     -�@     �@     ǰ@     ��@     ��@     ��@     ��@     ]�@     Ļ@     
�@    ���@     ;�@    ���@     7�@     ��@    ���@    � �@     T�@     ��@    �o�@     E�@    ���@    �?�@     e�@     ��@    �y�@    ���@     ��@    �e�@     ��@     ��@    ��@     �@    ���@    �$�@    ���@    �&�@     ��@     m�@     ��@     w�@     e�@     �@     V�@     8�@     ��@     N�@     ��@     ��@     ��@     N�@     �@     ̥@     �@     �@     v�@     b�@     h�@     B�@     ț@     \�@     ��@     �@     @�@     h�@     H�@     <�@     ē@     ��@     ��@     ܐ@     ��@     p�@     8�@     ��@     ��@     h�@     �@     ȇ@     Ј@     ��@     �@     x�@     x�@      �@     ��@     ��@     X�@     �@     ��@     @�@     P�@      �@     x�@     �@     �@     �@     �{@     p~@     @|@     �{@      {@     0|@     �y@     �u@     0w@      x@     t@      v@     `u@     �v@     �u@     �t@     �t@     t@     �q@     0r@     �w@     �u@      p@     pq@      n@     �v@     @s@     @o@     �n@     @o@      p@     �j@     �l@      l@      j@     �h@     �g@      g@     `e@     @e@     �b@     `g@      c@     @e@      c@     @c@     @b@     �a@     @c@     @_@     �`@     �]@     �`@     @[@      ]@      X@      ^@     �]@     @Y@     �[@     �X@     @Z@     �\@      Z@     �V@     @W@     �X@     @X@     @W@     @S@     �T@     �R@     @U@      T@     �S@     �V@     �N@      O@      L@     �R@     �S@     �O@     �Q@     �L@     �K@     �F@      E@     �M@      I@     �I@     �J@      I@     �P@     �D@      B@     �F@      E@      B@      >@     �C@     �F@     �G@      D@     �A@      <@      ?@      G@      ;@      :@      C@     �B@      @@      6@      ;@      6@      ?@      :@      >@      <@      9@      1@      7@      7@      2@      :@      @      3@      2@      2@      *@      *@      0@      1@      3@      &@      (@      3@      *@      .@      1@      ,@      (@      ,@      $@      *@      (@      "@       @      $@      "@      .@      ,@      @      @      @      &@      @      @      $@      &@      @      @      @      @      @      @      @      @      (@      @      @      @      "@       @      @      $@      @      @     �u@      w@      @      @      @      @      @      @      @      @      &@      $@       @      @       @       @      @      @      @       @      &@      .@      &@      @      @       @      @       @       @      @      0@      "@      @      @      &@      @      @      0@      &@      $@      &@      *@      ,@      (@      &@      2@      1@      .@      ,@      1@      3@      :@      ,@      4@      2@      7@      4@      2@      5@      5@      7@      =@      7@      9@      6@      5@      6@      4@     �C@      ?@      5@      =@      6@      A@      9@      B@      <@      @@      B@     �D@      E@      9@     �B@      H@     �D@     �G@      J@     �@@      E@      D@     �E@     �K@      L@     �N@     �K@     �M@     �M@     �Q@     �N@      L@      N@     �R@     �N@     �P@     @Q@     �Q@      T@     �T@     �T@     @S@     �R@     �S@     @S@     �W@     �P@     �X@     @X@      Z@     @Z@     �`@      ^@     ``@     �^@      `@     �\@     �a@      `@     `a@     �b@     �a@     �b@     �e@     �b@     @d@     `d@     �g@      f@      j@     @f@     �g@      i@      k@      j@      i@     �h@     `n@     �m@     �l@     `n@     Pp@     �l@     �q@     Pr@     Pr@     @p@     �r@     �r@      s@     �r@     �u@      t@     0u@     �s@     `w@     0x@     �v@     �v@     0x@     �y@     Pz@     }@     p|@      z@      {@     �}@     ��@     ��@      �@     (�@     �@     h�@     `�@      �@      �@     ��@     �@     X�@     0�@     ��@     8�@     0�@     x�@     �@     ��@     �@     H�@     ȋ@     <�@     �@     �@     ��@     8�@     ܑ@     �@     (�@     p�@     `�@     p�@     ܖ@     ��@     ȗ@     4�@     ��@      �@     X�@     `�@     �@      �@     4�@     �@     �@     b�@     ��@     ��@     |�@     �@     t�@     ��@     j�@     d�@     ^�@     ��@     ˰@     ֱ@     ��@     �@     ��@     ͷ@     ��@     �@     �@     �@    �3�@     ��@    ���@     ��@     \�@    ���@     #�@     D�@    �~�@    ���@     ��@     ��@    ���@     /�@    ���@     y�@    �W�@    �e�@    ���@     `�@    �r�@    ���@    �A�@    ��@    ���@     7�@     ��@     4�@     '�@     ��@     @�@     �@     B�@     ��@     %�@     �@     [�@     �@     �@     C�@     ��@     ��@     ��@     R�@     ��@     ��@     #�@     ͷ@     �@    �u�@     ԧ@     8�@     `@     �q@     �h@     �Y@     �E@      0@      @       @      $@      @        
�
predictions*�	   ���ӿ   �G�@     ί@!   �A;$�)�0��jO@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�>h�'��f�ʜ�7
������6�]�����[���FF�G �pz�w�7��})�l a�1��a˲?6�]��?>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@�DK��@{2�.��@�������:�              �?      �?      @       @      @      @       @      2@      2@      7@     �B@      G@      F@     �T@     @W@     �R@     �Z@      ]@     @X@     @]@      Y@     �]@     �\@     �\@     �\@      V@      Y@     �\@     �R@      V@      S@      O@     �P@      R@      I@     �H@      H@      ?@      =@      @@      5@      ?@      9@      1@      2@      ,@      *@      0@      $@      "@      "@      $@      @      @      $@      �?      @      @      @       @       @      @       @      @      @      @       @       @              @              @      �?      �?      �?      @      �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?      �?              �?       @      �?              @       @              @              @      �?      �?       @       @      @      @      @       @      �?      @      @      @      @      @      @      @      @      @      @      .@      @      &@      $@      0@      4@      "@      3@      .@      3@      .@      7@      3@      2@      3@      @@      =@      ;@      :@      <@     �B@      E@     �C@      :@     �A@      C@      B@      D@     �H@      9@      ;@      @@      B@      @@      E@      @@      ;@      .@      4@      9@      4@      8@      3@      (@      (@      1@      0@      "@      $@      @      @      .@       @      @      $@      @      @       @               @       @       @       @              �?       @      �?               @              �?        w��P4      9 ��	E�<n���A*�g

mean squared errorN)�<

	r-squared\>�>
�L
states*�L	   ����   �$�@    c2A!�	%[cT��)
����@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              ~@     L�@      �@     ��@     �@     ��@     �@     0�@     ��@     đ@     h�@     ��@     P�@     ��@     ��@     H�@     ��@     ��@     6�@     ��@     ��@     ر@     o�@     G�@      �@     ߰@     �@     �@     ��@     v�@     *�@     ��@     շ@     ��@     ׼@     �@    ���@    �!�@     ��@    ���@    �+�@     ��@    �U�@    �R�@    ���@     >�@     ��@    ��@    �O�@     ��@     ��@    ���@    ���@    �f�@     �@     ��@     ��@     \�@    �"�@    ���@     m�@     ��@     ��@    ��@     ��@     ��@     D�@     ��@     ��@     ��@     ű@     �@     ��@     ԫ@     �@     ,�@     x�@     x�@     �@     ʣ@     0�@     h�@     ܠ@     ^�@     2�@     0�@      �@     P�@     ��@     8�@     Ж@     ��@     H�@     Г@     ��@     ��@     ��@     ԑ@     ��@     x�@     (�@     (�@     Đ@     h�@     ��@     ؉@     �@     H�@     H�@     ��@     �@     (�@     ��@      �@     X�@     ��@     H�@     P�@     ��@     p�@     ��@     ��@     �@     �@     �{@     0~@     �z@     P{@     P{@     �{@      {@     �w@     @z@     @x@     `x@      w@     pv@     0v@     �s@      u@     �q@     �u@     �q@     �q@      s@     �q@      s@     �q@     �p@     �o@     �z@     �m@      m@     �n@      o@     �r@     0t@     @n@      e@     �j@      g@     �f@     `h@     �d@      f@      e@     `e@      b@     �d@     `d@     �b@      d@     �b@      a@     �]@     �_@     �c@      Z@     ``@      a@      ^@     �_@     �W@     @\@     @\@     @W@      [@     �X@      ]@     @Z@     �W@      W@      Y@     @[@     �R@     @W@      V@     @Y@     �S@     �T@      T@     @R@     �Q@     @T@      S@     @R@      O@     �M@      O@     �P@     �M@     @P@      M@     �J@     �L@      I@      J@      P@     �C@      J@      E@      G@      C@      ?@     �N@     �C@     �J@     �@@     �@@     �C@     �C@      E@     �C@      >@     �F@      E@      7@      B@      4@      6@      :@      :@      ;@     �@@      >@      =@      ;@      ?@      :@      "@      ,@      :@      9@      9@      ,@      1@      6@      0@      2@      ,@      0@      .@      *@      ,@      2@      0@      2@      *@      (@      (@      (@      &@      (@      $@      ,@      ,@      (@      &@      (@       @      @      &@      "@      @      @      "@      (@      $@      @      @       @      "@      @       @      (@      @      @      ,@      @      @      @      @     �v@     �x@       @      @      $@      @      @      @      $@      @       @      "@      @      @      "@      @      @      @      @      &@      @      $@      &@      �?      (@      @      $@      @      "@      $@       @      ,@      *@      @       @      $@      @      0@       @      5@      4@      ,@      "@      .@      "@      2@      ,@      .@      .@      *@      1@      1@      1@      6@      5@      $@      =@      3@      4@      >@      :@      0@      3@      6@      7@      5@      ?@      <@      6@      ?@      :@      :@      5@      :@      <@      A@     �C@      =@      A@      ?@     �B@      F@      @@      ?@      E@      E@      F@      D@     �K@     �D@      C@     �H@     �H@     �H@     �G@     @Q@      I@     �J@     @R@     �O@     @Q@      O@     �M@     �Q@     @Q@      M@     �Q@      V@     �U@     @S@     @S@     �T@     �W@     �X@     �U@     @[@      Z@     �\@      Y@     �\@     @_@     �Z@      ^@     @^@     �^@     �b@      `@      `@     `c@     �d@      a@     �e@     �c@      g@     �c@      e@     �g@     �g@     �h@     �f@     �f@     `k@     @m@     �i@     �l@      m@      l@     `n@     `p@     �n@      q@     pp@     �p@     0s@     �q@     �t@     �q@     �r@     @s@     �t@     @u@     �u@     v@     �t@     @v@     �v@     x@     pv@     px@     y@     |@     0|@     ؃@     �|@     �|@     P~@     �}@      �@     ��@     @�@     ȁ@     ��@     �@     (�@     X�@     �@     ��@     ��@     ��@     ��@     �@     �@     (�@     P�@     ��@     �@     `�@     4�@     0�@     ��@     ȏ@     ��@     4�@     ��@     �@     ̔@     �@     Ĕ@     �@     ĕ@     Ж@     d�@     ��@     ��@     `�@      �@     L�@     D�@     �@     ��@     \�@     ��@     ��@     ޣ@     �@     ֧@     ~�@     �@     �@     L�@     T�@     *�@     ߰@     ˱@     �@     ��@     ��@     ]�@     \�@     ��@     ��@     Ž@    �&�@    ��@     &�@     ��@     ��@     �@    �K�@     c�@     )�@    ���@     C�@    ���@     ��@     ��@    ���@    ���@    �$�@    ���@     i�@     ��@     �@    �%�@    ��@     `�@     ھ@     ��@     H�@     0�@     ��@     C�@     Y�@     8�@     ��@     �@     "�@     ��@     �@     S�@     �@     ��@     C�@     ��@     F�@     ��@     �@     ,�@     �@     ζ@     ¶@     ��@     ��@     |�@     Ѐ@     `t@     �k@     �a@      T@     �@@      "@      &@      &@      �?        
�
predictions*�	   ���׿   `��@     ί@!   �c{�?)g�I8�{N@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7���f�ʜ�7
������1��a˲���[���FF�G ����%ᾙѩ�-߾})�l a�>pz�w�7�>O�ʗ��>>�?�s��>6�]��?����?f�ʜ�7
?>h�'�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?��tM@w`<f@�DK��@{2�.��@�������:�               @               @      �?      �?       @       @      @      @      @      @      .@      2@      <@      7@     �@@      I@      H@      L@     �N@     �M@     @R@     �T@     �W@     �Y@     �]@      \@     �_@     @^@      Y@     @T@     @\@     �T@     @Y@     �R@      U@     �J@     �N@     �O@      K@      K@      I@      @@     �B@      B@     �C@      7@      8@      1@      =@      1@      ,@      ,@      ,@      ,@      .@      "@      @      @      "@      "@      "@      @      @       @      @      @      @      @      @      �?              �?      @      �?      @       @      @              �?      �?      �?      @               @      �?              �?              �?               @      �?              �?              �?              �?              �?              �?              �?               @      �?               @              �?      @      �?              �?       @      �?      �?      �?      �?      @      @      @      @      @      @       @      @      @      $@       @      @      @      @      "@      &@      (@      *@      0@      (@      ,@      5@      ,@      1@      1@      6@      >@      9@      9@      9@      4@      D@     �D@     �@@      A@     �A@     �C@      H@      B@      A@      B@      E@      H@      D@     �A@      :@     �A@      B@      ?@      9@      8@      9@      8@      6@      8@      "@      6@      4@      $@      2@      *@      0@      @      $@      &@      @      @      @      @       @      @      �?      @      @      �?      �?      @       @       @       @      �?              �?              @              @              �?        A�<�3      EN�	�Wn���A*�g

mean squared error�V =

	r-squared���>
�L
states*�L	   ����   ��|@    c2A!8�����)q��ޑ3�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             8�@     0�@     �@     x�@     ��@     Ȑ@     P�@     <�@     ،@     h�@     ��@     ��@     ��@     �@     �@     ��@     Ҵ@     ��@     ��@     w�@     ΰ@     ұ@     R�@     u�@     B�@     E�@     ��@     ��@     �@     d�@     ��@     �@     ��@     ��@     ��@     ��@     �@     ��@    �}�@     ��@    �~�@     ��@    ���@     ��@     ��@    �M�@     �@     ��@    �w�@    �	�@     ��@     ��@     ��@    ��@    ���@    ���@     R�@    ���@     ��@     2�@    ���@    ���@    ��@     �@    �m�@     ��@     �@     Y�@     �@     ��@     V�@     ��@     c�@     R�@     �@     ��@     t�@     �@     B�@     �@     @�@     l�@     f�@     n�@     �@     ��@     H�@     �@     h�@     ��@     `�@     ��@     ��@     h�@     0�@     �@     ��@      �@     0�@     $�@     ؐ@     x�@     ��@     ��@     h�@     ��@     �@     �@     Ї@     x�@     h�@     ��@     ��@     H�@     ��@     ؃@     (�@     p�@     Ѐ@     `�@     Ё@     ��@      @     (�@     �~@     �~@     p}@     �|@     �@     �~@     �|@     �{@     `y@     �y@     �y@     �w@     �x@     �w@     �v@     �u@     `v@     �t@     �t@     �q@     @r@     �q@     0p@     0p@     @s@     @q@     �{@     @p@     �o@      k@     `k@     �k@     �m@     �u@     `p@      h@     �i@      h@     @i@      h@      f@     �g@     �i@      h@      c@     �b@     �c@     �`@     �b@     ``@     �a@     �`@     ``@     `b@     @_@     �^@     �`@     �`@     �\@     �X@     `a@     @`@     @Z@     �Z@      Y@     �V@     @X@     @\@      ^@     �Y@     �U@     �X@     �U@     �U@     �[@      X@     �R@     �S@     �S@     �R@     @S@     �R@      S@     �O@     �Q@      Q@     �Q@     �Q@      I@     �N@      O@     �M@     �H@      J@     �L@      G@     @Q@      I@      G@      H@      A@     �J@      G@     �F@     �F@     �D@      F@     �D@      D@      F@     �@@      :@      B@      =@     �A@      9@     �@@      :@     �A@      7@      D@     �B@      7@      ?@      =@      3@      .@      5@      2@      3@      4@      .@      >@      4@      .@      4@      ,@      (@      4@      9@      *@      *@      3@      8@      2@      0@      7@      ,@      *@       @      .@      (@      (@      ,@       @      &@      $@      1@      &@      (@      "@      @      &@      $@      @       @      "@      "@      @      "@      "@      &@      @      @      $@              @      @     Py@     `w@      $@      @       @      $@      &@      @      @      @      @      @      @      @      $@      @      @      @      @      @       @      (@      "@      @      @       @      0@      $@      @      (@      "@       @      "@       @      @      (@      *@      *@      (@      5@      ,@      .@      2@      (@      3@      0@      (@      ,@      2@      2@      &@      3@      8@      .@      5@      3@      :@      ,@      =@      :@      9@      5@      8@      :@      5@      <@      >@      5@      8@      <@     �@@      A@      =@     �@@      D@     �B@      ?@      :@      9@      D@     �@@      E@     �A@     �A@     �D@     �F@      @@      I@     �H@     �D@     @P@     �N@     �D@      I@      B@     @P@      L@     �P@     �O@     �M@      O@     �K@      L@      Q@     @P@     �T@     @S@     �R@     @T@     �R@      T@     �X@     �T@     @Z@     �W@     �W@     �W@     @X@      \@     �[@     @_@     �a@     �]@     `a@     �a@     �a@     �`@     �`@     �c@     �d@     �b@     �g@      d@     �f@     �e@      h@      g@     �h@      j@     @i@     �i@      k@     �j@      l@     �m@     `m@     @m@     �p@      q@     �q@      q@     �p@     �r@     �r@     0s@     �p@     �r@      s@     �u@      u@     `w@     �u@     `u@     �u@     �x@      w@     �x@     `~@     {@     p|@     0x@     �}@     ��@     �}@     X�@     ��@     ��@      @     0�@     ��@     Ђ@     ��@     �@     (�@      �@     h�@     �@     `�@     ��@     X�@     0�@     8�@     ��@     `�@     �@     0�@     ،@     �@     l�@     `�@     В@     <�@     4�@      �@     X�@     �@     �@     L�@      �@     ��@     X�@     ��@     ��@     �@     �@     8�@     J�@     ��@     Ƞ@     Z�@     2�@     B�@     �@     �@     ��@     P�@     b�@     ^�@     �@     ��@     ��@     �@     C�@     ��@     _�@     ��@     ��@     �@     j�@     6�@     H�@    ���@    ���@    ���@     ��@    ���@     ��@     <�@     -�@     ��@    ��@    �{�@    ���@    �v�@    ���@    ���@     ��@    �l�@     A�@    �k�@    ���@    �B�@    ���@     �@     ��@     j�@     4�@     -�@     ׷@     ��@     �@     �@     Y�@     ۱@     ��@     ��@     ��@     ��@     ��@     �@     Z�@     ��@     �@     w�@     �@     '�@     ٵ@     -�@     R�@     ��@     ��@    �+�@     ��@     ��@     X�@     �v@     `o@     @d@     �`@     �D@      1@      &@      *@      @        
�
predictions*�	    ��տ   ���@     ί@!  \�w^@)lq2WQ@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
������6�]���>�?�s���O�ʗ����5�L�����]����I��P=�>��Zr[v�>>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�DK��@{2�.��@�������:�              �?              �?      �?      @      @      @      $@      @      &@      @      $@      "@      0@      7@      4@      3@      9@      9@      B@      ;@      ?@      ?@     �D@      C@     �C@     �F@     �F@      J@      E@     �B@     �E@      D@      C@      >@      B@      B@     �C@      >@      9@      *@      4@      =@      3@      *@      1@      $@      6@      ,@      (@       @      $@      "@       @      $@      @       @      &@      &@      @      $@      @      @       @      @      �?      @              @      @      @      �?      @      �?      �?      @      @       @              @      �?              �?       @      �?              �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              @      �?              @               @       @      �?      @      @       @               @      �?      @      @      @       @      @      @      @      @      *@      @      @      (@      "@      (@      @      (@      @      $@      0@      6@      ,@      .@      8@      7@      9@     �B@      ;@      C@     �E@     �F@     �G@      N@     �P@     �N@     �R@     �V@     �V@      \@      W@     @^@     @Z@     �\@      Y@     �W@     �]@     �X@      O@     �Q@     �T@      Q@     �K@      K@      H@      F@     �C@      >@      >@      ;@      6@      <@      8@      4@      5@      1@      &@      0@       @      @      $@      @      @      @      @      @      @      @      @      @      �?      @      �?      �?               @      �?              �?              �?        �`KB3      �%��	1�un���A*�f

mean squared error�{�<

	r-squared⑿>
�L
states*�L	   ���   ���@    c2A!C�;�q��)�Q�@c&�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             8�@     `�@     `�@     ��@     ��@     �@     x�@     ��@     ��@     ؋@     ��@     r�@    �\�@     ݲ@     i�@     �@     d�@     �@     ߰@     �@     |�@     ��@     /�@     گ@     �@     ��@     z�@     A�@     g�@     ��@     ��@     q�@     �@     '�@     �@     ��@     ��@    ���@     ��@     (�@    ��@     �@     �@    �>�@     ��@     �@     ��@    ���@    ���@    ���@    ���@     3�@    �G�@     `�@    ���@    ���@    �t�@     6�@    ���@     ��@    ���@    �>�@     ��@     ��@     �@     n�@     ˷@     ��@     5�@     H�@     �@     ��@     ��@     �@     ��@     j�@     ֧@     ڦ@     "�@     ��@     ̢@     �@     �@     �@     ��@     $�@      �@     (�@     <�@     0�@     ��@     �@     ��@     �@     ��@     ��@     �@     ��@     ؏@     <�@      �@     ��@     x�@     ��@     �@     �@     Ȉ@     ��@     ��@     x�@     8�@     ؅@     0�@     `�@     ��@     ��@     (�@     ��@     `�@     ��@     �@     ��@     �}@     �@     ~@     �~@     �}@     `{@     �y@     @y@     �{@     �|@     pz@     �y@      w@      w@     0x@     �u@     �v@     �u@      u@     w@     �r@      r@     �q@     �s@     �p@     0p@     Pr@     @p@     @y@     �r@     �o@     �l@     Pp@      s@     �s@     �l@     �l@     `k@     �f@     �h@     �h@     @h@     �e@     �h@     �f@     �e@     �e@     �c@     @b@     �c@      c@      c@      `@     �_@      b@     �`@     ``@     @^@     �]@     �a@     �[@     �\@     �`@     @]@     @X@     @_@      Y@      V@     �[@     �Y@     �]@     @\@     �U@     @V@     @U@     �V@     @Z@     �S@     �T@     @R@     �V@     �W@     @T@     �R@      N@      Q@     �M@      Y@      L@     �P@     �M@     �J@      E@      O@     �Q@     �M@      I@     �F@      I@     �H@     �J@     �F@     �G@     �F@      H@     �C@     �D@     �A@     �G@      B@     �A@      =@     �D@      ?@     �B@     �A@     �B@      7@      >@      <@     �@@     �B@      6@     �@@      B@      3@      <@      :@      6@      4@      7@      5@      4@      4@      :@      7@      0@      .@      :@      4@      8@      8@      1@      &@      (@      6@      .@      ,@      0@      6@      1@      0@      (@      @      @      $@      &@      $@      $@      (@      (@      ,@      .@      @      @      $@      (@      $@      .@      @      ,@      @       @      @      @      "@      @      @      ,@      @     �{@     �w@      @      @      @      $@      @      "@      @      $@      @      @      @      @      @      @      $@      @      @      @      $@      &@      @       @       @      0@      @      @      (@      0@      (@      @      .@      &@      3@      ,@      $@      &@      *@      .@      7@      0@      (@      ,@      1@      0@      ,@      .@      ,@      0@      :@      $@      2@      8@      ,@      3@     �@@      7@      2@      2@      1@      6@      7@      <@      <@      ?@     �A@      <@      =@      3@      >@      >@      ;@     �A@      :@      D@      @@      =@      A@      B@      B@      A@      @@     �H@     �@@      E@     �F@      G@     �G@     �H@     �I@      H@      B@     �E@      B@     �O@      O@     �P@      G@     �M@     �H@     @R@      N@     �Q@      R@     @U@     �Q@     �R@     �V@     @S@     �U@     �Y@     @\@     @^@     �Y@     @Z@     @\@     �X@     �Y@     @[@     �\@     �^@     �[@     �]@     �\@      b@     �`@     `c@     @c@      c@     @d@     @c@     �d@     �f@     �e@     �e@     �e@      i@     �h@     �i@     `h@     �k@     �j@     �l@     �p@     �o@     �i@     �o@     �o@     @n@     @q@     @q@     `r@     �q@     Ps@     �q@     Ps@     Pt@     pt@     @t@      u@     Pu@     x@     �v@     �x@     �{@     �x@     �x@     `z@     �w@     �y@     X�@     ��@     `~@     �@     �|@     Ѐ@     �|@     0�@     `�@     Ѐ@     @�@     p�@     x�@     ��@     8�@     h�@     `�@     ��@     H�@     ��@     ��@     �@     p�@     ��@     ��@     (�@     D�@     p�@     h�@     ��@     Đ@     x�@     Đ@     ��@     $�@     �@     D�@     0�@     �@     �@     p�@     @�@     �@     ��@     4�@     L�@     ��@     ��@     ��@     |�@     �@     :�@     ��@     >�@     ��@     �@     v�@     f�@     ��@     d�@     ��@     ��@     c�@     ��@     b�@     @�@     �@     ��@     �@     ��@     |�@    ���@     ��@     ��@     -�@    ���@    ���@    �2�@    �y�@    �O�@    ��@    ���@     ��@    �8�@     ��@    ���@     �@     L�@     ��@    �X�@    ��@    �7�@     ��@    ���@     ��@     u�@     ��@     B�@     �@     u�@     ��@     ��@     S�@     �@     ��@     ��@     l�@     ��@     ^�@     ��@     ��@     �@     5�@     �@     ��@     ^�@     Ƹ@     ,�@     ÷@     �@     7�@     ��@     ؖ@     ��@     �{@      r@     �n@     �d@     @X@      K@      ,@      9@      &@        
�
predictions*�	   @�8ڿ   �h�@     ί@!  �?�E�)}�6J��Q@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�f�ʜ�7
��������[���FF�G �>�?�s����MZ��K���u��gr��1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?��VlQ.?��bȬ�0?�u�w74?��%�V6?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@w`<f@�6v��@�DK��@{2�.��@�������:�               @              �?      �?      @      @      @      .@      .@      5@      <@      <@     �E@     �B@     �L@      L@     �P@     @S@     �U@     @V@     �^@     �Z@      \@      ]@     �d@     �a@     �b@     �`@     �a@     �\@      V@     �X@     @V@     @Q@      M@     �G@      G@      ?@      H@      :@     �@@      4@      1@      9@      3@      3@      *@      1@      ,@      &@      *@      &@      $@       @      @      @      @      @      @      @      @       @      @       @      @       @      @       @       @      @       @               @      �?               @              �?       @      �?      �?               @      �?      �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?              �?              @              �?      �?       @      �?      @       @      @      @       @      @      @       @       @      @      $@      @      @      $@      @       @      @      $@      (@      1@      &@      0@      ,@      0@      $@      9@      >@      >@      4@      >@      >@     �C@     �A@      >@     �@@     �F@     �@@      C@      <@     �A@     �F@     �@@     �B@      B@     �@@     �@@      =@      ;@      9@      .@      9@      6@      6@      &@      *@      &@      $@      (@      "@      "@      @      @      $@      @       @      @      @       @      @      @      @       @      @      �?       @       @      �?              �?      �?      �?              �?      �?      �?              �?        �-�&�2      ���C	w/�n���A*�e

mean squared error���<

	r-squaredN~�>
�L
states*�L	   @���    (�@    c2A!	ݍ��u��)�L�w��@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Ȋ@     �@     ��@     ��@     h�@     H�@     x�@     X�@     ��@     x�@     ��@     ��@     C�@     �@     ��@     ��@     G�@     -�@     �@     �@     L�@     ֯@     5�@     �@     ®@     ��@     �@     ��@     ��@     -�@     h�@     ��@     ��@     �@     ��@     s�@    �?�@     )�@     h�@     �@    �U�@     ��@     ��@    �v�@    �g�@     �@    �W�@    ���@    ���@    ��@    ���@     N�@    �F�@    �7�@     ��@    �,�@    �f�@    ��@    ���@    ���@     �@    ���@     ��@     ޾@     ��@     z�@     ��@     ��@     
�@     ��@     ��@     ��@     �@     ެ@     ,�@     ��@     ��@     ��@     F�@     \�@     �@     �@     Π@     ��@     ��@     Н@     �@     ��@     ��@     ��@     �@     Е@     Д@     ��@     �@     Ȕ@     ��@     А@     �@     $�@     �@     ��@     ��@     ��@     ؏@     Њ@     ��@     @�@     h�@      �@     X�@     ��@     ��@     h�@     ��@     x�@     x�@     x�@     `�@     �@     ��@      �@     0�@     @�@     �}@     �~@     P|@     @|@     �|@     @{@     �z@     �z@     py@     �y@     �y@     �x@      u@     �t@     px@     �u@      s@      v@     v@     �s@     `p@     �r@     �r@     Pr@     �q@     p@     `s@     �y@     �m@     �l@     @k@     v@     �l@     �m@     �j@     `i@     �h@     �h@     �c@     `i@     �g@     �e@      d@     �e@     @g@     �f@     `b@      d@     @\@      b@     �a@     �b@      c@     �b@      `@     �_@      _@      a@     @^@     @^@     �_@      [@     �_@     �[@     �Y@     �\@     @[@     �[@     �Z@     @Z@      [@     �X@     �V@     @W@     @[@     �V@     �Y@      V@     �R@     @Q@     @R@     �P@     @S@     �U@     �Q@     �Q@      N@     @Q@     �N@     �K@     �O@      L@      H@     �P@      P@      H@     �G@     �G@     �L@     �M@      J@     �E@      H@     �C@     �I@     �C@      >@     �B@     �C@      E@      D@     �B@     �G@      A@      C@      A@      ?@      >@      B@      E@      <@      G@      7@      @@     �@@      ;@      >@      3@      <@      7@      7@      7@      8@      1@      7@      3@      8@      .@      5@      4@      8@      6@      2@      &@      3@      $@      .@      1@      0@      1@      .@      $@      0@      .@      1@      (@      *@       @      *@      1@      $@      ,@       @      @      @      @      &@       @      $@      @      (@       @      (@       @      "@       @      $@       @     @     �w@       @      @      "@      @      @      @       @       @      @      $@      @      @      "@      "@      (@      @      (@      @       @      ,@      "@      &@      &@      @      @      (@      &@      *@      @      (@      0@      &@       @      2@      .@      *@      ,@      0@       @      5@      .@      7@      2@      0@      7@      2@      $@      0@      8@      "@      .@      1@      8@      9@      7@      6@      =@      =@      5@      :@      9@     �@@      9@      :@      ?@      5@      7@      E@      >@      A@      ;@      9@      8@      D@     �C@      D@     �G@      <@     �E@     �J@      F@      G@     �B@     �L@      J@      J@      E@     @P@     �G@     �D@     �M@     �J@      N@      P@     �K@      R@     �P@     @R@      L@      J@     �N@     �N@     �T@     �Q@     @V@     �R@      V@     �U@     �V@     @R@      [@     �Z@     �U@     �X@     @^@      [@     �]@      ]@     �^@     �\@      `@     @`@     �^@     @b@     �a@     �d@      d@     �d@     �d@      d@     `d@      f@     `e@     �g@     `h@      j@     �g@     �f@      m@      m@     @l@      k@      k@     �m@     �o@     �k@      p@     q@     �l@     q@     Pr@     r@     pr@     �s@      s@     �q@     Pu@     �v@     �y@     �t@     `x@     `x@     py@     @z@     �z@     0y@     �|@     ��@     �}@     �z@     �z@     ��@     �~@     �~@     �@     Ё@     �@     ؂@     �@     ؁@     0�@     ��@     ��@     `�@     �@     (�@     ؆@     ��@     ��@     ��@     �@     8�@     ��@     Ԑ@     ��@      �@     @�@     �@     ��@     D�@     `�@     ��@     <�@     $�@     �@     p�@     ��@     ��@     P�@     x�@     Ě@     ��@     �@     ��@     t�@     �@     ��@     (�@     ܢ@     x�@     �@     H�@     �@     �@     ^�@     ֪@     ��@     ��@     ��@     ��@     �@     ^�@     ��@     ��@     ��@     �@     ~�@     �@     w�@    ��@     ��@     
�@    ���@    � �@    ���@    ���@    ���@    �O�@     N�@    ���@    ��@    �s�@     �@     �@    ���@    ���@     ?�@     ��@    �b�@    �9�@    ���@     ^�@    �#�@    �/�@     ��@     %�@     v�@     ˸@     �@     8�@     c�@     y�@     ��@     O�@     �@     ˱@     ��@     D�@     ױ@     ��@     �@     -�@     l�@     z�@     T�@      �@     ��@     ��@     ��@     `�@     F�@     ��@     ��@     @�@     �z@     �n@     �h@     �c@     �]@     �B@      7@      &@        
�
predictions*�	   �.�ٿ    ��@     ί@!  ��t1@)����S@2�W�i�bۿ�^��h�ؿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��f�ʜ�7
������>�?�s���O�ʗ�����[�?1��a˲?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?{2�.��@!��v�@�������:�              �?              �?      �?       @      @      @      $@      @      ,@      "@      .@      9@      ;@      6@     �F@      H@     �P@      T@      U@     @[@     �Z@     �b@      f@     �g@     `c@      g@     �a@     �Y@     �Z@      W@      R@     �R@      S@     �M@      M@     �C@     �F@      6@      6@      9@      *@      ?@      >@      *@      *@      &@      .@      (@      $@      &@      @      @      @      @       @      @       @       @      @      @      @       @      �?       @      �?      �?              �?      @      �?       @      �?              �?      �?               @               @              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?              @              �?              �?      �?              @              @      @       @      @      �?      @      @      �?       @      @      @      @      @      @      @       @      &@      0@      @      *@      ,@      *@      &@      2@      ,@      2@      :@      :@      ;@      5@      >@      ;@      ;@      ?@      ?@     �B@      C@      H@     �F@      G@      E@     �D@      E@     �@@      @@     �E@      A@      B@      8@      <@      8@      >@      <@      9@      9@      5@      1@      1@      2@      0@      &@      ,@      &@       @       @      @      @      "@      @       @      @      @      @      @       @      �?      �?       @      @              �?       @       @              �?        {�ACr3      �i	9a�n���A*�f

mean squared errorD��<

	r-squared�P�>
�L
states*�L	   ����   ���@    c2A![	�����)%��y܌�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �@     `�@     �@     ��@     p�@     @�@     Љ@     P�@     ��@     ��@     H�@     ܟ@    ��@     `�@     ��@     ��@     M�@     /�@     ��@     ~�@     ��@     D�@     5�@     \�@     ��@     l�@     Ư@     6�@     �@     ݱ@     �@     �@     ��@     �@     ��@     o�@     ��@     ��@    ���@     �@     H�@     Y�@    ��@    ���@     (�@    ���@    ���@     ^�@    ���@     [�@     T�@     ��@    ���@    �S�@    ���@    ���@     ��@     ��@    �c�@    �I�@     ��@     ��@     3�@     -�@     ��@     <�@     ��@     &�@     y�@     B�@     f�@     ܰ@     خ@     ­@     �@     L�@     Ω@     �@     6�@     ֤@     �@     F�@     ��@     (�@     D�@     X�@     �@     ��@     `�@     H�@     ��@     ��@     ��@     �@     0�@     ��@     ȑ@     đ@     Џ@     T�@     x�@     ؍@     ��@     �@     P�@     ��@     (�@     �@     ��@     h�@     ��@     X�@      �@     ��@     ȅ@     (�@     ��@     @�@     P�@     ��@     0�@     ��@     0�@     �@     �}@     �}@     �|@     @}@     �{@     �z@     �y@     `{@     py@     z@     �u@     pu@     �u@     �v@     �t@     0w@     t@      t@     �r@     �s@      r@     �r@     �q@     �s@     pq@     �p@     �p@     �x@      o@      k@     @k@     z@     �p@     `m@      j@     �k@      l@      h@      g@     �i@     �e@      g@     �f@     �g@     �e@     �f@     �a@     �b@     `f@     �a@     `a@     �b@      c@     `a@     �b@     `a@     `a@      _@      ]@     @]@      ^@     @_@     �X@      Z@     �Y@     �^@      Z@      ^@     @W@      X@     @X@     �Y@     @X@     @U@      V@     �W@     @R@      Y@     �S@     �S@      T@     �Q@     �S@     �Q@     �T@      R@     �M@     �G@      N@      R@      Q@     �N@     �S@      N@     @Q@     �L@      K@      P@     �K@     �K@     �C@      G@      J@     �G@      I@     �I@     �H@      @@     �D@      D@      D@      G@     �C@     �C@      J@     �C@      D@     �D@     �A@      C@     �@@      ;@      B@      @@      5@     �A@      C@     �A@      @@      @@      .@      .@      4@      9@      4@      9@      :@      2@      0@      8@      8@      3@      3@      1@      4@      5@      1@      *@      .@      .@      4@      .@       @      &@      &@      .@       @      *@       @      0@      ,@      (@      "@      (@       @      *@      *@      (@      &@      @      "@      1@      (@      &@       @      @      *@      @     x�@     �w@       @      @      @      @      @       @      @      $@      @      @       @      @      "@      $@       @      "@       @      &@      @      &@      "@      ,@      (@      .@      &@      &@      *@      1@      $@      $@      "@      @      &@      &@      9@      *@      ,@      0@      >@      0@      2@      6@      @      5@      *@      9@      2@      ,@      9@      ;@      ,@      :@      >@      A@      0@      6@      <@      :@      9@      >@      ;@      9@      <@      >@      =@      ;@      =@     �A@      @@      B@      =@     �E@     �C@     �A@     �D@      H@      A@      A@      L@      F@     �F@      G@      D@      I@      J@      G@      L@     �E@     �F@     �I@     �G@      O@     �K@     �N@     �S@      K@     �P@     �S@     @P@     �R@     �Q@     @P@     �R@      T@     �U@      T@     �X@     @S@     �W@     �Y@     @V@     �\@     �X@      [@      W@     @]@     �Y@      _@     @_@     �]@      a@     �`@     �]@      a@     �`@     �a@     �`@     �b@     �f@     �d@     �d@     �e@     �i@     �g@     @e@     @j@     �i@      j@      j@     �h@     `m@     �l@     �j@     �n@     �n@      o@     �n@     �r@     �p@     �q@     r@     `s@     �r@      t@     �t@     t@     pt@     �u@     �u@     �v@     �x@     �v@     �{@     �w@      |@     (�@     @     �}@     p}@     �|@     �|@     p~@     �~@     �@     X�@     h�@     ��@     ��@     �@     ��@     8�@     �@     ��@     P�@     ��@     �@     ��@     ��@     H�@     �@     P�@     @�@     0�@     `�@     ��@     8�@     �@     ��@     ��@     �@     ��@     ��@     \�@     ��@     @�@      �@     D�@     @�@     ��@      �@     ��@     ,�@     �@     ��@     �@     П@     2�@     ��@     ҡ@     p�@     �@     8�@      �@     ��@     ��@     |�@     
�@     ��@     @�@     �@     ޱ@     q�@     ��@     մ@     �@     ��@     ��@     ��@     I�@    ��@    ���@     �@    �=�@     ��@    ���@      �@     ��@     =�@    ���@    ��@    ���@     }�@     5�@     ��@    ��@     ��@     �@    �0�@     �@    ��@     ��@     �@     C�@     2�@     4�@     Y�@     �@     ۵@     ��@     ��@     m�@     հ@     �@     5�@     ��@     ߰@     p�@     �@     α@     �@     g�@     ��@     0�@     ��@     �@     ��@     ��@     ��@     [�@    ���@     �@     �@      �@     �@     `�@      v@     pr@     �j@     @e@      `@      3@      ,@        
�
predictions*�	   @�Rڿ   ��	@     ί@!  @�i��)�S�1LuI@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�x?�x��>h�'��f�ʜ�7
������I��P=��pz�w�7��jqs&\�ѾK+�E��Ͼ��>M|K�>�_�T�l�>�FF�G ?��[�?6�]��?����?x?�x�?��d�r?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?S�Fi��?u�rʭ�@�DK��@�������:�              �?               @      �?              �?      �?              @      @      $@      ,@      (@      4@      >@      3@      >@      @@      7@      L@     �R@     �S@     @U@      X@     �`@     �b@     �c@     @d@     �e@     �c@     @`@     `b@     ``@     @Y@     �V@     �R@     @Q@     �Q@     �K@      C@     �C@     �E@     �D@     �@@      9@      9@      ,@      2@      1@      (@      3@      1@      "@      "@      (@      (@      @       @      @      @       @      @       @      @      �?      �?      �?      �?       @      @      �?      @               @              �?              �?      �?              �?       @               @               @              �?              �?              �?              �?              �?              �?               @              �?      �?              �?              @      �?               @      �?       @      @      �?      @      @      �?       @      @      @       @       @      @      @      @      @      @      @      @      @      @       @       @       @      *@      (@      .@      "@      3@      ;@      1@      8@      0@      4@      @@      ;@      =@     �C@     �@@      ?@      @@     �A@     �C@      D@     �A@      @@      C@      <@     �C@      A@      >@      9@      9@      7@      <@      0@      8@      7@      1@      (@      1@      0@      (@      $@      .@      $@      (@      @       @      @      @      @      @      $@      @      @      @      @      @      �?      @              @      @              �?              �?       @              �?        �/0β2      ���	_��n���A*�e

mean squared error�y�<

	r-squared���>
�L
states*�L	   `���   ��@    c2A!\���4��)��/ {�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     �@     ��@     (�@     �@     ��@     `�@     ��@     �@     ��@     ؐ@     Z�@    ��@     �@     `�@     �@     W�@     V�@     ��@     ��@     u�@     ��@     ǰ@     �@     �@     ��@     _�@     �@     z�@     �@     L�@     ��@     �@     �@     з@     ��@     �@     C�@     ��@    �[�@    ���@     ��@     `�@    �M�@     '�@    �9�@    ��@    �\�@      �@    ���@     ��@     w�@    ��@     %�@    �x�@    ��@     ��@     $�@     �@     ��@     ^�@     0�@    ���@    �M�@     ۽@     }�@     i�@      �@     �@     Ƴ@     l�@     �@     ��@     ��@     6�@     `�@     ֩@     ��@     8�@     l�@     n�@     F�@     >�@     Ԡ@     .�@     8�@     ��@     ��@     \�@     ��@     ��@     ؖ@     ��@     |�@     ��@     (�@     4�@     ��@     ��@     �@     �@     P�@     p�@     �@     0�@     \�@      �@     ��@     ؈@     ��@     ؇@     @�@     �@     ��@     �@     Є@     @�@     0�@     ��@     @�@     ��@     �@     (�@     (�@     �@     �~@     �@     P}@     �}@     �|@     �z@     `y@     �|@     �z@     �w@      x@     w@     �w@     @u@     �u@     �v@     �u@     �t@     0t@     @t@     �s@     �s@      s@     �r@     �r@     @s@     @z@     �o@     @n@      q@     �w@     �n@     �n@     @m@     `l@     �h@     �k@      i@     @h@     �g@     �h@      g@     `e@     `f@     �e@      i@     �h@     �a@     @f@      d@     `c@     @d@     �b@     �b@     �a@     @`@     @^@      a@     �c@     �^@     �`@      `@      ]@     @Y@     @]@      [@     �^@     �\@     �Y@     @]@     �W@      [@     �U@     �T@      X@      T@     �X@     @U@     �T@     @S@     �Y@     �T@     @V@     �W@     �S@     @S@     @S@     �K@     �O@      O@      P@     �M@     �O@      L@     �O@      M@     �K@     �P@      H@     �F@     �L@      =@     �D@      K@     �F@      H@      H@      E@      J@     �L@     �I@      E@      C@      E@      D@      A@      E@     �A@     �B@     �A@      A@     �B@      ;@      A@      5@      ;@      ?@     �B@      ;@      6@      8@      6@     �B@      5@      5@      6@      :@      <@      3@      :@      .@      *@      9@      .@      ,@      5@      1@      ?@      2@      1@      .@      0@      @      3@      1@      0@      4@      (@      (@      @      &@      "@      @      &@      "@      2@       @      ,@      &@      $@      $@      @      $@      *@      "@      ,@      @     ��@     �w@      @      $@      &@      @      @      &@      "@      "@       @      $@      $@      @      .@      @      &@      "@      @      *@       @      0@      ,@       @      *@      5@      2@      $@      ,@      *@      (@      6@      *@      &@      ,@      "@      ,@      .@      0@      1@      3@      *@      ,@      8@      ;@      1@      5@      1@      ,@      4@      .@      5@      <@      6@      6@      5@      4@     �B@      8@      9@      @@      4@      =@      =@     �A@      7@      ?@      B@      8@      ?@      E@      A@      ?@      5@      >@     �E@      @@      G@      C@      C@      D@      D@     �F@     �G@      J@      O@     �I@      I@     �I@      D@     �N@      N@     �J@      I@      N@      L@     �N@     �Q@     �P@      O@     �Q@      S@     @Q@     @P@      T@     �V@      Y@     �X@     @U@      V@     �T@     �X@     �Y@     �[@     �W@      ]@     �]@      \@     �Z@     �\@     �_@     @_@     �_@     @`@     �`@     `a@     `d@     �`@     @c@     �c@     @d@     `c@     �b@     �e@     �g@     �g@     `h@      j@     �h@     �k@     �g@     �k@     �k@     �k@     �k@     �n@     �o@     �p@     0p@     pq@     `p@     @q@     pu@     �q@      s@     �r@     @u@     �v@     �v@     �u@      w@     @u@     `w@     �x@     �z@     �y@     �y@     @|@     ؀@     P{@     �|@     �z@      }@     �@     �~@     �@     ��@     ��@     8�@     X�@     ��@     8�@     P�@     ��@     `�@     `�@     �@     ��@     ��@     p�@     ��@     �@     ��@     �@      �@     h�@     X�@     ��@     ��@     ��@     ȏ@     t�@     �@     4�@      �@     \�@     ��@     d�@     ��@     @�@     ��@     Ė@     ��@     ؘ@     D�@     Ԛ@     0�@     ��@     x�@     �@     ܡ@     ��@     ��@     ��@     ��@     ��@     ި@     ��@     &�@     ��@     ��@     $�@     T�@     ��@     �@     
�@     �@     ��@     >�@     л@     ��@    �g�@    ���@     �@     ��@    ���@    �H�@    �,�@    ���@    ���@     ��@     ��@     \�@    �8�@    ���@    �\�@     �@    �"�@    ���@    ���@    �8�@    ���@     ��@     �@    �E�@    ���@     d�@     ӻ@     �@     N�@     r�@     C�@     �@     n�@     |�@     ?�@     ��@     3�@     `�@     ��@     ��@     ʱ@     ��@     ��@     ��@     ��@     ��@     F�@     �@     �@     $�@    ���@     ��@     \�@     �@     ��@     ��@     @x@     �s@      f@     `e@     `c@      :@      .@        
�
predictions*�	    �ؿ   ���@     ί@!  d��T@)��QJ�Z@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��[^:��"��S�F !�ji6�9���T7����5�i}1��FF�G ?��[�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�6v��@h�5�@{2�.��@!��v�@�������:�              �?      @       @      �?      �?       @      @      $@      @       @      (@      7@      .@      <@     �C@     �F@      G@     �D@     �P@     @Q@      \@     �V@     �X@     �[@      _@     �[@     �[@     �[@      [@      ^@     �V@     �V@      Q@     �K@     @R@      L@      N@     �A@      E@     �B@      =@      9@      ,@      3@      2@      1@      *@      *@      "@      ,@      "@      @      .@      $@       @      @      @      @      @      �?      @              @      �?      �?      @       @               @      @       @       @       @      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?               @      �?      �?       @      �?      �?              �?       @               @      �?      �?      @      �?              @      @       @      @      &@      @      @      &@      @       @      @      @      .@      *@      .@      *@      1@      0@      7@      3@      3@      ?@      >@      8@      :@      >@      ;@      F@     �B@     �E@     �@@      I@      D@      C@      D@     �D@     �A@     �G@     �H@     �E@      C@      G@     �C@     �C@      9@      @@     �@@     �E@     �C@     �A@      ;@      @@     �@@      =@      =@      7@      ,@      3@      1@      ,@      ,@       @      ,@       @      $@       @      @      @      @       @      @       @      �?      �?      @      @      @      �?              �?              �?        ��VB3      �%��	�i�n���A*�f

mean squared errorp��<

	r-squared&��>
�L
states*�L	   ����   ���@    c2A!`�����)[>�93i�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     �@      �@     ؋@     ��@     ��@     �@     @�@     p�@      �@     H�@     ��@     ��@     u�@     ܳ@     ��@     ��@     H�@     _�@     ±@     ��@     �@     D�@     ��@     �@     2�@     Ʈ@     U�@     $�@     �@     ��@     ��@     ��@     ��@     ^�@     <�@     W�@     ��@     �@    ���@    �j�@     ��@    ���@    ��@     ��@     �@     ��@     ��@    � �@     ��@     ��@    ���@     ��@     =�@    ��@    ���@     %�@    ���@     �@     �@    �u�@    �8�@    ���@     �@     L�@     ;�@     #�@     ��@     �@     Y�@     �@     2�@     ��@     �@     �@     X�@     �@     ��@     ��@     @�@     �@     ��@     �@     Ȟ@     ĝ@      �@     ��@     ��@     $�@     ��@     H�@     @�@     �@     ��@     l�@     ��@     �@     �@     Ԑ@     ��@     `�@     X�@     ��@     ��@     H�@     ��@     �@     ��@     0�@     h�@     ��@     І@     ��@     ��@     0�@     ��@     ��@     `�@     ؁@      �@     ��@     ��@      �@     ��@     �@     ��@     ��@     @@     �@      }@     �{@     �|@      |@     �w@     `|@     �z@     �z@     `z@     Px@     Pw@      v@      v@     �s@     s@     ps@      s@     �t@     @u@     �s@      t@     �z@     �s@     �p@     �p@     q@     �x@      r@     @n@     `p@     �o@     �k@     @i@     �n@      m@     `k@     �h@     @i@     �f@     �g@      i@     �d@     �h@     `f@     �e@     `f@     �c@     `f@     �c@     `d@     @c@      a@     �b@     �a@     �b@     �_@     �_@      `@     �\@      ^@     @[@     @^@     @\@     �^@     @W@     �X@     �^@     @^@     �W@     �U@     �Z@     �Y@      X@     �S@     �V@     @X@     �U@     �X@      T@     @Q@     �V@     �V@     �P@      P@     �L@     �Q@      M@     @Q@      F@     �N@     �Q@     �O@      N@     �P@     �K@     �P@      G@     �G@     �I@     �H@      K@      B@      K@     @P@     �K@     �D@      J@      E@     �F@     �H@     �G@     �A@      >@      A@     �D@     �B@      :@      A@      @@      7@      C@      @@      8@      8@      9@      <@      9@     �A@      5@      6@      <@      0@      :@      <@      2@      0@      5@      5@      7@      8@      1@      $@      .@      9@      4@      2@      6@      (@      8@      *@      5@      5@      4@      .@      *@       @      &@      (@      $@       @       @      "@      ,@      3@      "@      .@      1@      .@       @      0@      $@       @      .@     ��@     �y@      @      &@      @      $@      &@      @      "@      *@      "@      @      @      @      0@       @       @      &@      "@      @      ,@      0@      (@      @      "@      @      *@      &@      (@      3@      *@      8@      0@      5@      .@      6@      3@      2@      4@      3@      6@      4@      &@      *@      0@      6@      8@      4@      1@      4@      6@      6@      1@      4@      <@      7@      6@      4@      5@      ;@      ;@      7@      :@      >@      >@      ;@      9@     �B@     �D@      B@      D@      B@     �A@      C@     �B@      D@     �C@      @@     �@@      A@     �H@      E@     �F@      E@     �J@     �I@      J@      J@     �K@      K@      K@     �O@     �O@     �Q@     �O@     �M@     �N@     �R@      N@     �N@     @U@      P@     �Q@     �V@     �V@     @X@     @V@     �U@     �X@     @U@     �U@     �U@     �Y@      W@     �]@     �X@      \@     �\@     �^@     �_@     �\@     @^@     �`@     �`@      a@      `@     @b@      d@      d@      c@     �b@     `e@     �i@     `f@     �e@      f@     @i@      g@     `f@     @e@      i@     �i@     �m@     �j@     �k@     `l@      m@     p@      o@     �r@     r@     �p@     �p@     0q@     0s@     `r@     ps@     �u@     u@     �u@     w@     �u@     �u@     u@     Px@     Px@     h�@     �}@     �|@     �{@     �{@     @     p{@     �|@     �~@     0~@     ��@     �@     @�@     �@     �@     ��@     ��@     �@     ��@     H�@      �@     ��@     ��@     P�@     x�@     ��@     H�@     �@     ��@     ��@     ��@     ��@     ȏ@     ��@     ��@     ��@     ��@     (�@     4�@     �@     l�@     �@     ��@     �@     �@     Ė@     X�@     x�@     @�@     ��@     l�@     $�@     p�@     �@     ��@     �@     ��@     "�@     ��@     ��@     ަ@     ��@     h�@     ��@     ƫ@     �@     ��@     ��@     ��@     ޳@     ��@     =�@     F�@     ��@     W�@     ��@    ���@     ��@    ���@     '�@     ��@    �Z�@    �(�@     !�@    ���@     ��@    ���@    �Y�@     ��@     ��@     ��@     %�@    ��@     ��@    ���@     [�@     ��@     ��@    �>�@    ���@     ��@     N�@     ��@     Ʒ@     �@     .�@     !�@     f�@     �@     �@     ��@     �@     V�@     ��@     q�@     h�@     9�@     '�@     %�@     ɳ@     �@     Ʒ@     ƻ@     �@     ��@    ���@     ̩@     ��@     ��@     8�@     (�@     �@     `u@      u@      j@     �n@     @`@      (@        
�
predictions*�	   �Lڿ   �ɑ@     ί@!  dw�iL�)+.o�kJ@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��>h�'��I��P=��pz�w�7��jqs&\��>��~]�[�>1��a˲?6�]��?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?�E̟���?yL�����?{2�.��@!��v�@�������:�              �?      �?              �?      @       @       @      $@      (@      (@      1@      :@      C@     �@@     �G@      H@     �J@     @Q@     �S@     �T@     �T@     �]@     @\@     �^@     `b@     �b@     �d@     �`@      a@     �d@      _@      `@     �]@      V@     @U@      U@     �R@      M@     �H@     �H@     �D@      D@     �@@      B@      9@      2@      1@      $@      7@      2@      (@      $@      3@      $@      &@      (@      @      @      @       @      �?      @      @              @      @      �?      @       @       @      �?       @               @      @      �?      �?               @              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              @              �?              �?               @      �?               @       @      �?      @              �?       @       @      �?      @      @       @      �?      @       @      @      @       @      @      "@      @       @      "@      1@      @      $@      (@       @      0@      2@      1@      1@      9@      .@      8@      ;@      =@      9@      5@      >@      =@      5@      =@      4@      <@      8@      .@      (@      1@      6@      4@      5@      $@      ;@      2@      1@      (@      $@       @       @      *@      .@      .@      "@      ,@      $@      &@      @      $@      @      @       @       @      @       @      @      @      @      @      @      @       @      @       @      @              �?              �?              �?        Jl&��3      �3�	z�o���A*�g

mean squared errorZ�<

	r-squared��>
�L
states*�L	   ����   �7�@    c2A!��N����)t"�CΥ�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              �@     `�@     `@     �@     ��@     ԑ@      �@     �@     ��@     Ќ@     ��@     �@    ���@     �@     :�@     3�@     Բ@     ��@     @�@     ��@     $�@     "�@     Z�@     ��@     T�@     ­@     k�@     	�@     ��@     `�@     ٱ@     T�@     �@     V�@     ٷ@     �@     W�@    ��@     �@    �e�@     1�@     C�@    ���@    �v�@     l�@    ���@    ���@    ���@    ��@     ��@    �{�@     G�@     ��@     �@    �X�@    ���@    �p�@    ���@     ��@     �@     ��@     !�@     &�@     ��@     j�@     +�@     ��@     ��@     ׵@     ��@     ��@     ı@     w�@     C�@     <�@     ~�@     8�@     ��@     <�@     V�@     6�@     4�@     ��@     ��@     ��@     ��@     ��@      �@     8�@     ��@     ؗ@     ��@     ��@     X�@     |�@     ��@     H�@     Б@     ��@     ��@     �@     P�@     ��@     �@     p�@     �@     h�@     (�@     H�@     ��@     @�@     �@     0�@     �@     �@      �@     8�@     0�@     ��@     Ѕ@     H�@     Ё@      �@      �@     ��@     ��@     @�@     p~@     �@      ~@     �~@      |@     �@     �@     �z@     `z@     {@     @z@      y@     y@     �w@     w@     �x@     �u@     v@     �v@     `u@     0u@     pt@     �u@     �|@     �s@     �q@     0r@     Pp@     �q@     `{@     Pr@     �m@     0p@     �o@     �l@      n@      n@     �h@     �k@      j@     `m@     �k@     @i@     �i@      j@     �d@     @g@     �f@      g@     �e@     �h@      b@     `d@     �e@     `d@     @f@      f@      e@      `@     �b@     @`@     @[@     �`@      a@     �a@      `@     �`@      b@     �^@     �Z@     �^@     @Z@      [@     @Z@     �W@      U@     @V@      W@      V@      T@     �U@      U@     �R@     �R@     @V@      R@     @Q@     �U@      Q@     �R@     �Q@      S@      G@     @Q@      S@     �L@     �J@      K@      J@     �H@     �J@     �P@     �G@      L@      J@     �H@     �J@     �N@      H@      J@      Q@     �E@     �E@      =@      G@     �E@      I@     �C@     �A@      E@      E@      >@      =@     �B@      =@      B@      9@      @@      =@      A@      ?@      <@      :@      ?@      8@     �@@      >@      4@      9@      8@      4@      8@      5@      :@      7@      2@      4@      =@      0@      0@      5@      4@      5@      (@      ,@      *@      $@       @      1@      (@      $@      .@       @      $@      @      &@      (@      ,@      &@       @      $@      $@      &@      @      1@     ��@     �|@      $@      @      @      @       @      "@      @      @      .@      (@      (@      1@      *@       @       @      @      (@      @      (@      "@      .@      &@      .@      &@      ,@      "@      2@      1@      3@      (@      3@      *@      *@      .@      1@      $@      1@      7@      :@      =@      7@      1@      (@      :@      4@      4@      *@      8@      8@      6@      7@      7@      (@      4@      3@      ;@      B@      >@      1@      >@      9@      2@      B@     �A@      ;@      A@      A@      @@     �A@     �C@     �D@      ?@      @@     �@@      F@      I@      C@     �D@     �D@     �I@     �L@     �A@     �I@      F@      G@      I@     �P@     �I@     �K@     @P@      R@     �P@     �S@     �M@     �Q@     �R@     @S@     @R@     �P@     �U@     �S@     @Q@     �X@     �X@     �V@      Y@     �Y@     �W@      Z@      V@     �V@      \@     �\@      [@     �X@     @\@     �^@     �_@      ^@      a@     �b@     �`@     �b@      b@      f@     �b@     �b@     @b@     `e@     �d@     �e@     �g@     �e@      g@      j@      g@     �k@     @i@      k@     �k@      m@     �k@     �n@     @n@     �o@     Pp@     �l@     Pr@     �p@     �s@     �q@     Ps@     �s@     s@     pu@     �s@     u@     �t@      w@     `w@     �w@     �v@     {@     ��@     �x@     Px@     �|@     �}@     �{@     �{@     �@     `|@     �@     `~@     ��@     �@     ��@      �@     ��@     ��@     0�@     ��@     0�@     p�@     ��@     `�@     ��@     �@      �@     ��@      �@     @�@     �@     h�@     �@     ��@     ��@      �@     x�@     �@     Ē@     ��@     T�@     ��@     @�@     D�@     �@     Ȕ@     |�@     ��@     ��@     �@     X�@     \�@     ��@     D�@     ��@     �@     �@     �@     ¢@     n�@     ^�@     ��@     ��@     ԫ@     h�@     @�@     H�@     1�@     �@     N�@     ĳ@     !�@     ��@     -�@     ��@     �@     ��@    �|�@    ���@    �_�@    ���@     ��@    ���@    ���@    �r�@     �@    ��@    ���@    ���@     �@    �i�@     q�@     ��@    ���@    ���@     ��@    �w�@     ��@     b�@     ��@     �@    ���@     8�@     Թ@     �@     �@     ��@     &�@     o�@     ��@     �@     .�@     ��@     3�@     ��@     Y�@     S�@     x�@     �@     +�@     �@     k�@     ڵ@     �@     ��@     �@     (�@    ��@     �@     �@     ��@     ��@     ��@      }@     �w@     �r@     @h@     �j@     `c@      3@        
�
predictions*�	   @�ڿ    m�@     ί@!  ��z�9@)���PE�M@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]����FF�G �>�?�s���})�l a��ߊ4F��})�l a�>pz�w�7�>I��P=�>��Zr[v�>f�ʜ�7
?>h�'�?x?�x�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?u�rʭ�@�DK��@�������:�              �?               @      @      �?              @      @      @      @      2@      (@      5@      3@      ;@      E@      E@     �E@      G@     @P@      O@     �N@      P@     �S@     @V@     �V@     �V@     �W@     @Y@     @T@     �P@      T@     �S@     �R@     �K@      K@      M@     �I@      H@      A@      C@      E@      <@      7@      :@      9@      5@      6@      4@      4@      &@      1@      1@      &@      .@      @      $@      $@      @      @      @      @      @      @      @      @       @      �?       @      @       @      @      @              @              @      �?      �?       @       @              @      �?      �?      �?      �?      �?              �?              �?              �?      �?              �?              �?              �?               @               @       @              @       @       @       @              �?       @              @      @      �?      @              �?      @      @      @      @      @      @       @       @      @      "@      @      $@      "@      "@      0@      ,@      &@      "@      1@      ,@      ,@      .@      7@     �A@      ?@      ;@     �C@      D@     �H@      D@      E@      H@      K@     �R@      I@     �P@     �S@     @T@      J@     �Q@     @R@     �D@     �K@      F@      H@      C@      8@      :@      <@     �@@      6@      @@      5@      3@      3@      5@      :@      .@      2@      2@       @       @       @       @      (@      @      @      "@      �?      @      @      @       @      �?      @      @      @              @      �?      �?       @       @              �?              �?        ��34      9 ��	�o���A*�g

mean squared errorB��<

	r-squared���>
�L
states*�L	   ����   ��@    c2A!
�~�jo��)g�p@	�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     ��@     P�@     ��@     p�@     ��@     �@     X�@     �@     ��@     h�@     (�@    ��@     +�@     ߳@     ȶ@     ��@     ��@     /�@     ��@     ��@     ��@     <�@     ��@     �@     D�@     �@     p�@     l�@     �@     \�@     ��@     ӳ@     B�@     *�@     *�@     ��@     ��@     ��@     ��@    �1�@     ��@    �S�@    �K�@    �-�@     ��@     ��@    ���@     ��@    ���@     �@    ���@    �}�@     ��@      �@     q�@     ��@     ��@     M�@    ���@    �`�@     ��@     C�@    ���@     ��@     ��@     ��@     ��@     Ե@     ��@     ��@     ��@     ��@     p�@     ��@     ��@     ��@     ��@     Ħ@     ��@     J�@     v�@     ġ@     �@     D�@     8�@     ��@     К@     ̚@     P�@     ؙ@      �@     D�@     �@      �@     Ԕ@     �@     l�@     ��@     D�@     L�@     Џ@     ԑ@     �@     ��@     p�@     ��@     ��@     ��@     8�@     ��@     x�@     ��@      �@     ��@     ��@     h�@     �@     �@     ��@     ��@     ��@     ��@     �@     p�@     �@     @�@     �}@     �}@     `�@     `}@     �}@      |@     �}@     P~@     �z@      z@     �z@     (�@     Px@     @x@     @z@     �v@     pv@     �u@     Pv@     �t@     �u@     �t@     p{@      t@     ps@     �s@     s@     �q@     0r@     �|@     �s@     0p@     �q@      l@     �o@     �p@     �m@     �l@      l@     �k@     @j@     `i@     �h@      i@     �i@      g@     �i@     �h@     �h@     �d@     @e@     �e@     �e@     �g@     @g@      d@     @c@      e@     �`@      b@     �`@     @d@     �`@     �Y@     @b@     �`@      `@     @[@     �[@     �]@     �W@      Y@     @^@     �]@     �W@     �X@     @^@      X@     �W@     @]@     @V@     �S@     @V@     �V@     �T@     �W@     @U@     �S@     �S@     @P@     �Q@     �M@     �Q@      Q@     �J@      O@     �K@     @R@     �P@      I@     �H@     �I@     �E@     �L@     �J@      M@     �E@      G@     �I@      N@     �M@     �F@     �C@     �D@     �I@     �C@     �F@     �E@     �A@     �A@     �C@      B@     �D@      C@      G@      ?@      ;@      8@     �A@      B@     �A@      <@      A@      ;@      8@      ,@     �C@      <@      6@     �A@      ;@      3@      <@      ,@      5@      >@      5@      5@      5@      1@      7@      1@      0@      3@      3@      0@      .@      *@      3@      &@      *@      3@      @      0@      1@      *@      2@      "@      @      (@      "@      1@      "@      "@      $@     @�@     �}@      @      "@      @      *@      "@      $@       @      @       @      @      @       @      @      *@      $@       @      "@      &@      &@      @      (@      &@      2@      $@      *@      ,@      *@      6@      .@      4@      0@      "@      1@      ,@      (@      .@      0@      1@      &@      5@      4@      5@      5@     �A@      5@      <@      5@      5@      9@      8@      9@      4@      4@      7@     �@@      2@      ;@      1@      ;@      ;@      @@     �A@      8@      :@      9@      :@      ;@     �B@      >@      A@      =@     �C@      B@      E@      I@     �G@     �J@      ?@      H@     �B@      F@     �A@      G@     �G@     �I@      G@     �F@      G@     �L@      S@      H@      N@     �N@     �P@     @Q@     �L@     �V@      R@      S@      O@     �T@      U@      U@     @S@      X@     �Y@     �W@     �U@     �W@     �S@     �Z@     �U@     �]@     �[@     @]@     �\@     @^@     @_@      ]@     �]@     �\@     �]@     �_@      `@     �a@      c@     `d@     @d@      b@      c@     @b@     `e@     �c@     �g@      d@     �f@     `h@     `l@      j@     �l@      i@     �k@     �n@      m@     `l@     �p@     @p@     Pq@     �p@     Pp@     @q@      s@     �r@      s@     �u@     �r@     0s@     �t@      v@     �u@     �u@     �t@     `|@     �}@      x@     �{@     �v@     `z@     �{@     p�@     p~@      �@     �}@     `~@     ��@     ��@     �@     �@     ؁@     ��@     ؃@     ��@     �@     ��@     ��@     X�@     ��@     x�@     X�@     ��@     �@     x�@     ��@     ��@     H�@     X�@     ��@     �@     ��@     ��@     ��@     ��@     H�@     �@     �@     \�@     �@     T�@     L�@     Е@     ��@     ��@     |�@     4�@     \�@     T�@     Ğ@     $�@     $�@     ��@     ��@     ��@     *�@     ֧@     �@     ̩@     ��@     l�@     |�@     ��@     ϱ@     �@     ��@     d�@     A�@     h�@     ��@     !�@     ]�@    �:�@    ���@     `�@    �$�@     b�@    �%�@    ���@     ��@     �@     _�@    �F�@    ���@     ��@     =�@    �~�@     ,�@     g�@    �~�@     ��@    �y�@     ��@     ��@     �@     �@    ���@     j�@     +�@     n�@     ��@     ��@     �@     k�@     �@     A�@     �@     ��@     r�@     >�@     �@     6�@     ʱ@     Z�@     _�@     ��@     ��@     ܵ@     �@     ��@     �@     ��@    ���@     >�@     �@     �@     �@     ��@     �z@     `|@     �m@     �g@     �q@     �B@      2@        
�
predictions*�	   �߿    �@     ί@!  p���&@){�D��S@2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�DK��@{2�.��@�������:�              �?      �?      @              �?      �?      @      "@      @      &@      "@      *@      1@      :@      9@      :@      D@     �B@      K@     �K@      S@     �U@      O@      V@     �U@     �X@     �Y@     �V@     @V@      [@      Z@     �]@     �Y@     �T@      V@      Q@     @S@     @P@     �M@      L@      M@      E@     �A@     �D@      C@     �@@      ;@      4@      4@      4@      2@      0@      0@      .@      *@      *@      "@      @      $@      $@      (@      @      @      @       @      @      @       @      @      �?      @      @       @              �?       @              @       @      @              �?      �?      @              �?      �?              �?              �?      �?              �?              �?      �?              �?              �?      �?      �?               @               @              �?              @       @              �?              @               @               @      @      �?      �?      @               @       @      @      @      @      @      @      @      @      @      @      "@       @      ,@      0@      (@      0@      2@      7@      .@      ;@      6@      1@      ?@      5@      8@      ?@      3@      >@      @@      C@     �E@      C@      B@     �E@      C@      F@      D@      B@     �C@     �B@      H@      =@     �A@      9@      B@      ;@      9@     �B@      C@      @@      <@      6@      7@      5@      0@      ,@      .@      2@      ,@      $@      "@      *@      @      @      @      @       @       @      @      @      @      @      �?       @              @              @              �?      �?       @              �?        g��>r3      �i	2W8o���A*�f

mean squared error;��<

	r-squared��>
�L
states*�L	   @���   �u�@    c2A!D�Ǭ�<��)������@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             L�@     ؄@     �@     �@     ��@     l�@     ��@     �@      �@     H�@     ��@     V�@    ���@     F�@     �@     ݶ@     o�@     ��@     q�@     j�@     2�@     f�@     Ʈ@     v�@     ��@     ܮ@     ԯ@     ��@     �@     >�@     t�@      �@     �@     C�@     ��@     u�@     μ@    �%�@    �^�@    �A�@     ��@     S�@    �,�@    �	�@    ���@     ��@    �!�@    �Y�@     ��@     ,�@    ���@    �.�@     ��@     %�@    ���@     ��@     
�@    ���@     ��@    �n�@    �>�@    ���@     ��@    �Z�@     Z�@     !�@     ��@     A�@     ��@     _�@     ��@     �@     ��@     |�@     �@     N�@     $�@     �@     Φ@     ֥@     :�@     h�@     ��@     Z�@     ��@     ��@     �@      �@     ��@     ��@     @�@     ؘ@     D�@     x�@     ��@     ��@     ĕ@     ��@      �@     P�@     �@     ̕@     ��@     ��@     Ѝ@     x�@     ,�@     ��@     (�@     ��@     �@     ��@     ��@     (�@     �@     �@     �@     �@     ��@     ��@     0�@     �@     ��@     X�@     X�@     h�@     8�@     ��@     �@     �@     �@     p}@     @{@     @~@     �~@     �{@     �{@     Ђ@     0z@     x@     �x@     px@     `x@     Px@     �v@     �v@     0u@      u@     �t@     �~@     @t@     pq@     0r@      t@     �s@     �q@      u@     `x@     �r@     pp@     �q@     �k@     �n@      p@     `k@     �l@      k@      k@      l@      m@      k@     �m@     �j@      e@      i@      h@     �h@     `c@     `f@     �b@     �e@     �f@     �d@     �b@     �e@     ``@     @`@      c@     @a@     `b@     �b@     �c@      ^@     @b@      `@     @[@      ]@     @^@      a@      \@      [@     �[@     �[@     @V@      Z@      S@      T@     �S@     �T@      T@     �W@     @V@     �Q@     �S@     @S@     @Q@     �U@     �S@     @U@     �P@     �R@      L@      R@      R@     �P@      S@     �J@     �L@     �K@     �P@      J@     �J@      N@     �K@     �L@      J@     �I@     �J@     �I@      K@      G@      F@     �E@     �I@     �D@      A@     �E@     �E@      D@     �E@     �B@     �G@     �D@      9@     �B@      ?@      ?@     �A@      B@      ;@      9@      D@      <@      8@      <@     �@@     �A@      ?@      4@      4@      1@      :@      >@      1@      >@      5@      ;@      *@      2@      :@      1@      1@      1@      5@      *@      9@      2@      2@      3@      ,@      4@      6@      4@      $@      1@      3@      0@      .@      (@      1@      &@      $@     ��@      }@       @       @       @      $@       @      @      @      @      *@      "@      &@      "@      &@      @       @      $@      "@      "@      1@      *@       @      2@      *@      0@       @      1@       @      "@      *@      4@      3@      .@      2@      5@      .@      0@      1@      5@      .@      5@      1@      0@      4@      9@      6@      *@      6@      2@      <@      2@      9@      @@      5@      ;@      2@      9@      5@      ;@      <@      5@     �A@      <@      @@      >@      :@      =@      @@     �C@      ;@      A@      >@      E@      @@      >@     �@@     �C@     �C@     �F@     �D@     �I@      J@      F@     �G@     �K@     �G@      K@      H@     �L@     �M@      F@     �N@     @Q@      N@     @R@      N@     �M@     �O@      P@      R@     �O@     �S@     @R@      R@     �U@     �U@      W@     �W@     @T@     @S@     �U@      W@     �X@     �X@     �Y@     �]@     @_@     �\@      _@     @Z@      `@     �`@     �a@     �_@      a@      b@     �_@     �d@     �a@      a@     @e@     �e@      b@     `e@     �c@     `k@     `e@      j@     �h@      j@     �j@      i@      h@      l@     �i@     �q@     @m@     �n@     �o@     pp@     0p@     @p@     �s@     @r@     �r@     �s@     pq@     Pt@      u@     �s@      w@     �{@     0{@     0x@     pz@     �x@     y@     �x@     �x@     `{@     �x@     ��@     `~@      }@     X�@      @     `~@     0�@     x�@     �@      �@     �@     �@     ��@     8�@     �@     ��@     �@     @�@      �@     X�@     H�@     8�@     ȉ@     ��@     ȏ@     �@     ��@     Ȏ@     ؎@     �@     |�@     ��@     0�@     ��@     t�@     ��@     ��@      �@     (�@     ��@     ��@     P�@     P�@     ��@     T�@     ,�@     $�@     ��@     |�@     ��@     �@     ��@     �@     ��@     
�@     ث@     �@     ��@     ��@     ��@     Y�@     ��@     ��@     ��@     F�@     p�@     O�@     �@     ��@      �@     ��@    ��@     F�@     y�@    ���@    ���@    �#�@     .�@     A�@     i�@     Y�@     ��@     ��@     P�@    �$�@    �U�@    �]�@    ���@    ���@    �o�@     "�@    ���@     6�@     ?�@     �@     Z�@     ��@     ��@     G�@     �@     ��@     b�@     �@     {�@     N�@     L�@     ΰ@     �@     u�@     �@     �@     ��@     ��@     P�@     q�@     ŷ@     ʼ@     �@     ��@    ���@     B�@     ��@      �@     ��@     `�@     ��@     Pu@     �j@     @h@     �k@      3@      *@        
�
predictions*�	   �Ѽ�   @�@     ί@!  ��(� @)�o��W\S@2�\l�9⿰1%���Z%�޿W�i�bۿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����[���FF�G �>�?�s���O�ʗ����h���`�8K�ߝ뾋h���`�>�ߊ4F��>��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?��tM@w`<f@!��v�@زv�5f@�������:�              �?              �?               @      @      @      @      @      @       @      "@      2@     �A@      >@      7@     �E@     �E@     �F@     �K@      Q@     �L@     �Q@     @S@     �T@     @V@      W@     �X@     �V@     �Y@      X@      Z@     �\@      X@      T@     �S@     �R@     @R@     �L@     @T@     �J@     �K@     �D@      A@      E@      A@      :@      6@      8@      6@      7@      .@      ,@      $@      (@      @      ,@      @      @      @       @      &@      @       @       @      @      @      �?       @      �?              @      @      �?      @      @              �?      �?       @      @      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @       @      �?      �?      �?      @       @      @      @      @      @       @      @      @      @      @      @       @      @      @      &@      @      @      @      @      *@      @      0@      3@      7@      .@      .@      3@      0@      2@      ;@     �B@      9@      <@      <@     �B@     �C@     �G@      >@      C@     �G@     �D@     �H@     �F@      L@     �G@      9@      G@     �D@     �F@     �B@      9@     �@@      6@     �C@      >@      >@      4@      0@      *@      0@      *@      2@      0@       @      .@      ,@      $@      "@      .@      @      @      @      @      @      @      @       @      @              @              @      @      @       @      �?              �?              �?        �3�kB3      �%��	��Ro���A*�f

mean squared error4L�<

	r-squaredp�>
�L
states*�L	   ����    �@    c2A!D1'sN��)$A��Ll�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     0�@     8�@     `�@     �@     ��@     Ȋ@     �@     8�@     8�@     ��@     �@    �e�@     K�@     M�@     ��@     �@     j�@     N�@     ��@     �@     %�@     �@     �@     ��@     2�@     *�@     ��@     Y�@     i�@     ް@     ��@     �@     ��@     ��@     �@    ��@     ��@     ��@    �)�@     �@     ��@    �	�@     ��@     ��@     ��@    ���@     ��@     �@     ��@    ���@     b�@     P�@    ���@    �'�@    �-�@     ��@     5�@     �@     ��@    ���@    ���@     �@    ���@     �@     �@     "�@     _�@     õ@     @�@     :�@     [�@     o�@     �@     ܬ@     &�@     �@     ʩ@     �@     ��@     �@      �@     D�@     ԡ@     r�@     �@     P�@     �@     ܛ@     P�@     ��@      �@     �@     ��@     �@     ȓ@     �@     0�@     ȑ@     8�@     8�@      �@     ��@     ��@     ��@     Ȍ@     h�@     Њ@     0�@     ��@     ȉ@     Љ@     h�@     ��@     ��@     P�@     ��@     8�@      �@     x�@     ��@     x�@     (�@     `�@     X�@     8�@     P�@     ��@     p@     �@     �@     Ѐ@      @      |@     �|@     P�@     �|@     �x@     �y@     �w@     �x@      y@     �v@     �y@     �x@      v@      u@     w@     0u@     �u@     �{@     �x@     t@     �q@     �r@     �q@      p@     p|@     p@     �o@      o@     �n@     �o@     �p@     �l@     `n@     p@     `k@      l@     �h@     �j@     `h@     �i@      j@     @g@     `g@     `j@     �e@     �g@      g@     �d@     �c@     �e@     `c@     �b@     `a@      e@     �b@     `b@      c@      c@      `@      ]@     @_@     �]@     @]@     �`@      `@     �^@     @\@     �]@     @Z@      Y@     �V@     �Z@     �Y@     �W@     @W@     �W@     �W@     �T@     �S@     �U@     �S@     �U@     �R@     @W@     �P@      M@     @Q@     �P@     @Q@     �O@     @Q@      R@     @T@     �P@     �O@      P@      P@      I@     �E@     �N@     �O@      I@     �H@     �Q@      E@      L@      E@      A@     �C@     �D@      E@     �L@     �I@     �A@     �D@     �H@      ?@     �G@     �@@     �B@      G@      @@     �D@     �C@     �A@      =@      @@      C@      6@     �A@      B@     �A@      8@      :@      ?@      ?@      8@     �A@      <@      8@      7@      9@      3@      7@      4@      4@      8@      .@      7@      5@      2@      7@      .@      8@      8@      4@      7@      2@      9@      *@       @      $@      *@      4@      1@      &@      *@      ,@      &@     �@     �}@      @      @      "@      @       @      "@      "@      @      "@      &@      (@       @      @      (@      &@      "@       @      $@       @      $@      &@       @      ,@      4@      (@      1@      ,@      "@      4@      0@      1@      *@      4@      *@      4@      "@      4@      6@      .@      5@      4@      ;@      7@      5@      &@      <@      ,@      :@      4@      4@      *@      (@      0@      6@      3@      9@     �D@      7@      :@     �A@      <@      8@      >@      <@      C@      >@     �A@      @@      E@     �@@      B@      1@     �C@      A@     �C@      D@      B@      G@     �F@     �L@      G@     �F@     �H@      G@      G@      K@      I@      O@     �J@     �N@     �L@      J@      L@      M@      J@     �O@      O@     �R@     �R@      Q@     @U@     @Q@      S@     �S@     @T@     �R@     @R@      U@      W@     �U@      R@     �V@     �[@     �X@     @]@     ``@     �Z@      ]@     �W@      ^@     �\@     @\@     @b@     ``@     @^@     �d@     @d@      c@     �c@     �d@     �c@      e@     �e@     `g@     �e@     @g@     �g@      g@     @h@     `k@     �k@     �j@     @k@     �i@     �k@      m@     �p@     @n@      r@      p@     Pp@     0p@     �r@     `p@     �r@      s@     �u@     �s@     �v@     �{@     @u@     t@     �v@     �v@     @w@      y@     0x@     `w@     pz@     Pz@     0@     �{@     �@     �|@     �@     ��@     �~@     P�@     Ȃ@     ؀@     ��@     (�@     (�@     ��@     ��@     0�@     H�@     H�@     ��@     @�@     (�@     @�@     �@     ��@     `�@     H�@     ��@     ��@     8�@     ��@     �@     ��@     h�@     �@     8�@     t�@     ؓ@     |�@     �@     `�@     ��@     x�@     ��@     �@     L�@     t�@     ��@     ��@     ֠@     ��@     
�@     �@     z�@     X�@     ��@     f�@     v�@     d�@     ��@     ��@     f�@     ��@     ��@     ��@     �@     ��@     ڹ@     �@     ��@     ��@     ��@     ��@     ��@     U�@     %�@    ���@    ���@    �S�@     A�@     ��@     %�@     ��@    �f�@     ��@    ���@     E�@    �c�@    ��@     L�@     ��@     ?�@     p�@     h�@     s�@     L�@     ��@     %�@     շ@     �@     M�@     �@     �@     �@     �@     ��@     İ@     ְ@     8�@     ��@     ΰ@     �@     ޱ@     �@     ��@     �@      �@     C�@     �@     Q�@     l�@     �@     X�@     �@     ��@     p@     �@     Pt@     �k@     �h@      l@      ;@      0@        
�
predictions*�	    �5�    2�@     ί@!  d��@�)Oxc�>V@2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����d�r�x?�x��E��a�W�>�ѩ�-�>�f����>��(���>f�ʜ�7
?>h�'�?x?�x�?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@�6v��@{2�.��@!��v�@�������:�              @       @              �?       @      @      �?      @      .@      3@      ,@      =@      B@     �@@     �F@      L@      O@     �N@     �O@      R@     �R@     �W@     �T@     �Z@     �Y@     �_@     ``@     �]@     �`@     @a@     �Z@      \@     @W@     �V@     @U@      V@     �S@     @Q@     �K@      N@     �G@     �H@     �B@      A@      7@      =@      7@      3@      1@      7@      ,@      *@      $@      $@      0@      "@      @      @      *@      @      @      @      @      @       @      �?      @      @      @       @      �?      @      �?      @       @               @       @      �?      �?               @      �?              �?       @              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?               @      �?       @      �?      �?       @      @      @      @      @      @      �?      @       @      @      $@      @      @      $@      @      $@      (@      $@      @      &@      $@      (@      1@      1@      ,@      (@      1@      7@      0@      5@      >@      5@      7@      8@      >@      9@      ?@      5@      5@      5@      :@      <@      <@     �@@      9@      ;@      8@     �A@      3@      5@      ?@      >@      6@      4@      0@      1@      2@      4@      2@      2@      0@      @      @      @      (@      @      (@       @      @      @      @      @      @      �?      @       @       @      �?      @              @              �?      �?               @      �?              �?        b-H
�3      �=%�	Q0po���A *�g

mean squared errorA~�<

	r-squaredz��>
�L
states*�L	    ���   ���@    c2A!qB����)la�ھD A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             0�@     ��@     Ј@     t�@     h�@     (�@     ��@     (�@     ��@     8�@     |�@     ��@     T�@     3�@     E�@     �@     ��@     ��@     �@     I�@     ٰ@     0�@     ��@     ��@     R�@     ��@     |�@     m�@     B�@     �@     ӱ@     ��@     ��@     ��@     0�@     ��@     �@     ��@    ���@    �^�@     q�@     g�@    ���@    ���@     ^�@     ��@    �)�@    ���@     ��@     ��@    ���@     ��@    ���@    �c�@    ���@    ���@     ��@     ��@    ��@    �%�@     H�@     ��@    �%�@    �r�@     �@     ٻ@     ��@     ;�@     D�@     �@     ��@     ߱@     ��@     ��@     ��@     ��@     0�@      �@     ��@     �@     �@     �@     ��@     ��@     ��@     :�@     L�@     l�@     ��@     ��@     ��@     t�@     ��@     ��@     �@     $�@     ��@     ��@     ��@     D�@     �@     ؓ@     ��@     ��@     X�@     ��@      �@     p�@     ,�@     �@      �@     X�@     ��@     �@     x�@     H�@     P�@     �@     ��@     x�@     Ȃ@     ��@     ��@     x�@     ��@     ��@     (�@     @�@      �@     ��@     ؀@     �}@     �{@     ��@      ~@     {@     p|@     �x@     �y@     �z@     �|@      u@     �z@     �x@     �w@     pu@     �v@     �v@     �t@     �v@     �q@     Pt@     �}@      t@     @r@     `q@     �q@     �p@      s@     �p@     �p@     �x@     p@     �k@     �m@     �n@     0p@      j@     �i@     �o@     �h@      j@      j@     �h@     �h@      j@     �e@     �d@     `h@      h@     �e@      e@      f@     @d@     @d@     �c@     �`@     �b@     @d@      b@      a@     �b@      `@     @_@     @^@     �_@     �a@     @b@     @Z@     @]@     �V@      [@      ^@     @Y@      Z@     @U@      W@     �Y@     �X@      V@     �W@     �W@      X@     @X@     �V@     @U@     @S@     �P@     �T@     �S@     �T@     �O@     �O@     �S@      S@     @Q@     �Q@     �Q@      O@     @Q@      N@     @P@      I@      N@      O@     �J@     �I@     �P@     �J@     �F@     �G@      K@     �D@      K@      M@     �H@      A@     �G@     �E@      @@     �G@     �E@     �@@     �A@      @@      E@      E@     �A@     �@@      ?@      B@      >@      B@      >@      =@      ?@     �@@      9@     �B@      ?@      7@      @@      ;@      A@      4@      <@      9@      2@      ;@      7@      3@      3@      ?@      3@      6@      :@      3@      5@      ;@      0@      ,@      6@      2@      0@      3@      2@      7@      .@      ,@      "@      1@      "@     Ȋ@     p�@      @      @      @      @      0@      (@      @      @      @      @       @      &@      @      $@      ,@      *@      .@      3@      (@      (@      ,@      *@      2@      $@      *@      1@      ,@      ,@      *@      ,@      ,@      0@      4@      3@      3@      0@      4@      2@      (@      2@      :@      1@      5@      <@      7@      ,@      0@      6@      <@      ,@      6@      ;@      8@      <@      :@      :@      2@      <@      ;@      D@      6@      ?@     �@@     �B@      =@      9@     �B@     �A@      A@     �B@      D@      <@      B@     �D@     �@@      B@      E@      B@     �D@     �H@      E@     �G@      P@      M@     �E@     �H@     �M@      G@     @P@      P@      P@      N@     @Q@      O@     �P@     �Q@     �G@     @P@      S@      T@      P@     �W@      T@      X@      O@     �V@      U@      V@     �Q@     �X@     @[@     @Y@     @V@     �Z@     �Y@     �W@     @^@     �[@      \@     �_@     @`@     �]@     �]@     �a@      a@     �a@      b@     @d@     �a@      c@      e@     �c@     `g@      g@     `f@     �g@     �i@     �h@      j@     �h@     �m@     �k@     `m@     �n@     `k@     �o@     �p@     �q@     �r@     �n@     pp@     �p@      q@     �q@      s@     �s@     @t@     Ps@     0s@     �t@     �|@     �{@     �v@     w@     �w@     �x@      y@     �{@     0z@     �|@     `�@     �|@     ��@     �@     �@     �@     ��@     `�@     ��@     ��@     ��@     �@     ��@     x�@     0�@     (�@     `�@     0�@     ؇@     ��@     ��@     ��@     ��@     (�@     ��@     ��@     Ȍ@     H�@     x�@     X�@     8�@     d�@     ��@     �@     l�@     ̔@     `�@     �@     L�@     ��@     |�@     t�@     0�@     P�@     ��@     ��@     �@     ��@     �@     @�@     ��@     f�@     �@      �@     ت@     ��@     �@     6�@     �@     �@     H�@     [�@     ��@     �@     m�@     `�@     ��@     [�@     �@     �@     ��@     ��@     I�@     ��@    ���@     �@    �M�@     ��@     {�@    ���@     %�@    ���@    �G�@    ���@    �p�@     ��@     ��@     L�@     ��@    ���@    ���@     q�@     2�@     #�@    �[�@     d�@     ڷ@     ��@     ͵@     |�@     ��@     ��@     3�@     *�@     i�@     ��@     ��@     .�@     8�@     ��@     o�@     �@     ��@     ��@     v�@     ��@     Q�@     c�@     X�@    �k�@     ֯@     (�@     ��@     ��@     0�@     ��@      v@     �j@     @k@     @n@      8@      0@        
�
predictions*�	   �z�ۿ   ���@     ί@!  ���:@),���\R@2���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������>�?�s���O�ʗ�����>M|Kվ��~]�[Ӿ����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?ܔ�.�u�?�DK��@{2�.��@�������:�              �?      �?      �?      �?       @      �?      @      @      @      @      @      .@      4@      4@      =@      ;@      ;@      A@      F@      I@      F@     �N@     �U@      S@     �T@     @]@     @Z@     @[@     �Z@      Z@     �Z@     �Z@     �Y@      T@     �T@     �R@      S@      V@      H@     �K@      H@      G@      I@     �A@     �A@     �E@      C@      @@      5@      9@      0@      3@      6@      0@      &@      .@      (@      (@      &@      @      @      $@      @      @      @      @      @      @      �?      @      @       @      �?      �?      @       @              �?              �?      �?              �?              �?      @              �?              �?              �?              �?              �?              �?      �?       @              �?      �?              �?      @      �?              �?      �?      �?      �?      �?              @      �?       @      @      �?      @      �?      @       @              @      @      @      @      @      @      @      $@      "@      $@      *@      "@      *@      ,@      &@      &@      8@      *@      .@     �@@      =@      .@      4@      ;@      ?@      6@      =@     �@@      B@     �@@      C@     �D@      G@      A@     �B@     �D@      =@     �B@      D@      C@      ;@      ;@      6@      9@      =@      =@      B@      9@      5@      =@      D@      ,@      ?@      5@      &@      ,@      5@      *@      ,@      ,@      &@      (@      "@      @      @      �?       @      @      @      @      @      @      �?       @       @      @       @               @      �?              �?        _�.B4      8�� 	xD�o���A!*�h

mean squared errorm��<

	r-squaredzk�>
�L
states*�L	   @���   �&�@    c2A!�m�n
d��)���<� A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Ԓ@     ��@     ��@      �@     ��@     ؉@     ��@     h�@     ��@     �@     �@     ^�@     ��@     �@     ��@     �@     �@     S�@     H�@     �@     ��@     �@     ��@     �@     ��@     N�@     S�@     H�@     J�@     l�@     ��@     8�@     1�@     �@     )�@     ��@     ��@    ���@    ���@    ��@     ��@     ��@    ���@    ���@     �@    ��@    ���@    ���@     ��@    ���@    �e�@    �h�@     ��@     ��@     �@    ���@     ��@     _�@     :�@     l�@     ��@     ��@    �
�@    �9�@     �@     9�@     ��@     �@     Ѷ@     ]�@     j�@     q�@     [�@     j�@     F�@     ج@     6�@     D�@     �@     �@     ֤@     ��@     Ģ@     ��@     ~�@     V�@     6�@     ��@     $�@     T�@     `�@     X�@     ��@     h�@     H�@     ��@     P�@     <�@     ��@     (�@     ��@     ��@     (�@     \�@     Ў@      �@     ��@     ��@     h�@     8�@     ��@     ��@     Љ@     (�@     ؅@     h�@      �@     ��@     ��@     Ѓ@     P�@     h�@     8�@      �@     �@     ��@     P�@     �@     �@     ��@     ��@     �@     `}@     �}@     �|@     �}@     �|@      {@     z@     `{@     �x@      w@     �v@     �w@     0v@     �x@     �u@     pt@     �t@     �u@     �t@     `t@      v@      s@     }@      q@     p@      q@     `n@     �p@     @r@     @p@      o@     �p@     `n@     `y@     pp@      m@      h@     �l@     �h@      i@     �g@     @h@     @i@      i@     @g@     �e@      h@      g@     �e@      h@      e@      c@      a@     �c@      c@     �a@     �a@     @e@     �b@     @d@     @`@     `c@     �b@      `@     �Z@     �`@     @^@     �Z@      \@     �[@      Z@      [@     @_@      S@     �W@     �Y@     @X@     �V@     �W@     �U@     �Y@     �V@     �T@     @V@     �R@      [@     �S@      R@     �O@     �R@      N@      R@     @S@     @R@      M@     �P@     �L@     �O@      M@      S@      Q@     �R@     @P@      I@      H@      J@     @P@      I@     �E@     �M@     �M@      G@      E@     �I@     �C@     �E@     �D@     �M@      F@     �E@      E@      @@     �G@      B@      D@      E@      @@      A@      F@     �C@      D@      A@      :@      ;@      >@      ?@      <@      ?@      3@      4@      3@      9@      3@      8@      8@      :@      9@      7@      9@      6@      0@      ;@      .@      7@      2@      2@      (@      2@      1@      5@      3@      .@      ;@      *@      5@      &@      $@      1@      .@      *@     �@     @�@      $@       @      $@      3@      @       @      &@      @       @      *@      @      "@      &@      &@      @      ,@      &@      @      $@      1@      @      1@      *@      7@      *@      ;@      1@      1@      &@      2@      2@      (@      ,@      ,@      2@      <@      0@      4@      "@      4@      5@      5@      :@      6@      ,@      7@      7@      5@      7@      6@      2@      =@      <@      8@      ;@      <@      :@      <@      :@      7@      <@      <@      9@      @@     �@@      >@      <@      B@     �D@      <@      @@      ;@      =@     �E@      C@     �D@     �D@     �H@     �I@     �K@     �D@      E@     �I@     �M@     �L@      N@      K@     �R@      K@      L@      J@     @P@     �O@     �P@     �I@     �P@     �H@     �R@     �J@      Q@     �R@     �P@     @U@     �R@      S@     �V@      V@     �U@      W@     �X@     �X@     �V@     �X@     �W@     �Z@     @[@      `@      [@     @Z@      _@     �_@     �a@     �_@     @a@     �b@     �a@     `b@      d@      e@      b@     �b@     @f@     @f@     �c@     @e@      g@     �g@      g@     `i@      j@     �l@     �m@     @o@     �n@     @h@      o@     `o@      o@      o@     �r@     0q@      n@     �r@     �q@     �s@     �r@     �u@     �u@     �u@     ps@     �u@     �}@      y@     �w@     �x@      y@     0x@     @{@     @z@      {@      }@     �}@     ~@     �@     �@     ��@     ��@      �@     Ё@     x�@     ��@     8�@     �@     ؆@     p�@     �@     �@     @�@     ��@     Ї@     @�@     �@     ȉ@     Ќ@     ��@     ��@     p�@     ��@     h�@     8�@     ̐@     d�@     ��@     ��@     ��@     ��@     8�@     �@     ��@     0�@     L�@      �@     ��@     �@     8�@     ��@     l�@     �@     �@     4�@     8�@     ��@     "�@     @�@     ��@     ��@     v�@     ��@     *�@     �@     a�@     <�@     ˴@     Z�@     ��@     ;�@     Һ@     Y�@     I�@     <�@     ��@    �}�@    ���@    ���@     ��@     ��@     �@    ���@    �"�@    ���@     ��@    ���@    ���@    �J�@    �#�@    �\�@    ��@     _�@     ��@    ���@     ��@    ���@    ���@     ��@     T�@     �@     b�@     -�@     U�@     �@     ��@     B�@     �@     u�@     b�@     �@     �@     U�@     ��@     ^�@     >�@     ��@     �@     s�@     ȵ@     D�@     �@     `�@     ��@     ��@     �@     0�@     (�@     h�@     ��@     ��@     �v@      i@     �g@     pq@      4@      3@        
�
predictions*�	    ɱ�    �I@     ί@!  ���M@)�J����T@2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ�x?�x��>h�'��>�?�s���O�ʗ���8K�ߝ�a�Ϭ(�E��a�Wܾ�iD*L�پ['�?��>K+�E���>�ߊ4F��>})�l a�>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@{2�.��@!��v�@�������:�               @              �?      �?       @              �?      @       @      @      *@       @      @      1@      @@      6@      <@     �@@      A@     �@@      M@      H@     �C@      G@      O@     �M@      P@     @Q@     �R@     �N@     �P@      U@      N@      J@     �R@     �N@     �R@     �P@      H@     �K@     �G@     �G@     �D@      E@     �A@      @@      ;@      <@      .@      0@      9@      .@      3@      6@      4@      1@      $@      @      @      &@       @      @      @      @      "@      @      @      @      @      @      �?      �?              @       @      @      �?      @      �?              �?              �?              �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @              �?      �?              �?               @      �?      �?      @      �?      @      @      �?       @       @      @      �?      &@              @      �?      "@      @      @      @      @      @      @      $@      $@      $@       @      @      $@      (@      ,@      9@      4@       @      >@      A@      C@      8@      6@      >@      A@      J@      J@      F@     �G@      K@     �G@     �M@     �P@     @P@      R@     �R@     @Q@     @Q@     �O@      Q@     �H@     �D@      E@      L@     �F@     �A@     �C@     �A@      ?@      8@      B@      6@      @@      ;@      1@      1@      3@      7@      ,@      .@       @      "@       @      @       @      @      @      "@      �?      @       @      @      @      @      �?      @      �?      �?               @               @              �?        �׳�3      �gw	+�o���A"*�g

mean squared error$��<

	r-squared��>
�L
states*�L	    ���   `e�@    c2A!�>C  ��)�x��G�@2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ؑ@     ȃ@     h�@      �@     (�@     p�@     ؄@     ��@     ��@     �@     �@     ��@     �@     ɳ@     ��@     Ŷ@     �@     q�@     ��@     H�@     ��@     ~�@     _�@     f�@     ԯ@     \�@     @�@     ��@     ��@     ��@     ��@     W�@     ��@     �@     ��@     t�@    ��@     D�@     d�@    ���@     ��@     ��@    �/�@    �
�@    ���@     U�@    ���@    �1�@    ���@    ���@     ��@    ��@     ��@    ���@    �>�@     ��@    ���@    ���@     ��@     ��@     o�@     ��@    �3�@     ��@     ��@     
�@     Ѹ@     o�@     /�@     ��@     �@     ��@     f�@     Ҭ@     ��@     ��@     ��@     ��@     ̥@     ��@     L�@     �@     ��@     ڠ@     L�@     ��@     ğ@     ��@     ��@     ܜ@     L�@     ��@     ȗ@     t�@     d�@     ��@     ��@     ��@     ��@     \�@     ��@     ��@     Ȓ@     ȍ@     ��@     ��@     ȋ@     �@     x�@     h�@     ��@     ��@     ȉ@     ��@     ȍ@     ��@     @�@     �@     ��@     Є@     x�@     ��@     ��@     `�@     h�@     ȁ@     p�@     ��@     `~@     ��@     H�@     ��@      �@     `�@     �}@     �z@     z@     {@     �z@     �{@     �x@     �y@      w@     �w@     pt@      v@     @u@     `t@     `u@     Pv@     0u@     �r@     t@      r@     @r@     Pt@      y@     �o@      o@      r@     @r@     p@     p@     P{@     �p@     �m@     �k@     �k@     �m@     �i@     `j@     `l@     �j@      l@     �g@      f@     �k@     �h@     @f@      h@      h@     `e@     @e@     @e@      e@      b@     �d@     �b@      a@     @d@     �a@      b@     �a@     @`@     �a@     @`@      ]@     @`@      _@     �[@      b@     �Y@     @Z@     �[@     �[@     �Y@     @V@     �Z@     �T@     �Y@      Z@     �X@      V@     �W@     �T@     �U@     @T@     �T@     @P@     �Q@     �S@     �U@     @S@     �R@      T@      P@      S@     @R@     @Q@     �R@     �O@      K@     �Q@      L@      G@      P@      P@     �I@      J@     �J@      H@      E@      F@      J@      G@      C@      E@     �D@     �F@      D@     �G@      D@     �D@     �F@      =@      B@      ?@     �G@     �D@      C@     �@@      C@     �A@      ?@      <@      <@      @@      5@      :@      4@      <@      @@      @@      6@      8@      6@      B@      .@      ?@      8@      >@     �@@      1@      2@      4@      *@      5@      1@      1@      9@      5@      *@      3@      5@      1@      $@      1@      0@      7@      &@      1@      ,@     X�@     ��@      @       @      $@      "@      $@      @      @      @      3@      ,@      "@      "@      $@      *@       @      *@      "@      @      *@       @      "@      5@      7@      "@      .@      .@      3@      0@       @      5@      1@      7@      1@      2@      1@      2@      3@      5@      (@      7@      4@      1@      8@      6@      4@      (@      9@      4@      4@      &@      0@      *@      4@      :@      7@     �A@      2@      B@      :@      =@      C@      ;@      <@      4@      A@      =@      D@     �C@     �C@      >@      9@      @@      A@      ?@      A@     �D@     �D@      F@      D@      P@     �J@      N@     �H@      H@      J@      N@      G@      O@      O@      K@      P@      M@     @P@     �M@     @S@     �P@     �Q@     �N@      R@     �P@     �R@     �H@      S@     �T@     �U@     �L@     �V@     �U@     �W@      [@     @W@     �Z@      [@     �Z@     �[@      [@     @\@      Z@     @Z@     �]@     �`@      _@     �^@     @a@     �^@      a@     �a@      b@      e@     `a@     `d@     �b@     `e@     @d@     �e@     �h@      k@      i@     �g@     �j@     �j@     �i@     `g@      m@     p@      o@     �l@     �p@     p@     Pq@     `o@     0q@     0r@     `r@     `p@     Pr@      s@     Ps@     `s@     �v@     �v@     �t@     @u@     �~@     Pz@     �y@     �v@      y@     �z@     z@     �|@     �{@      @     ��@     �~@     P�@     x�@      �@     ��@      �@     ��@     ��@     p�@     (�@     8�@     ��@     �@     h�@     x�@     8�@     H�@     �@     ��@     x�@      �@     ��@     ��@     (�@     ��@     ��@     ��@     ؏@     �@     �@     �@     ��@     ��@     �@     ��@     ��@     �@     ��@     ��@     h�@     @�@     P�@     |�@     ��@     ��@     ��@     �@     �@     *�@     X�@     D�@     4�@     ^�@     ,�@     ��@     Z�@     Ʊ@     �@     �@     ��@     ܵ@     з@     �@     D�@     ��@     >�@     %�@     M�@    ���@    �_�@     ��@     y�@     Y�@    ���@    ���@    �5�@     !�@     ��@     @�@     J�@     �@    ��@     ��@    ���@    �f�@     ��@     ��@     ��@     K�@    ��@     r�@     �@     ĺ@     [�@     S�@     ��@     �@     -�@     5�@     N�@     �@     ��@     o�@     ��@     ǰ@     �@     ��@     ��@     f�@     ��@     Q�@     �@     !�@     �@     ��@     m�@     ��@      �@     ��@     ��@      �@     �}@     �z@      m@     �i@      r@      ;@      3@        
�
predictions*�	   @6��   �6,@     ί@!  ��-�)�$�{�Y@2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ�6�]���1��a˲��ߊ4F��>})�l a�>pz�w�7�>I��P=�>x?�x�?��d�r?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@{2�.��@!��v�@�������:�              �?               @              @      @      @      @      (@      ,@      1@      5@     �A@      E@     �H@      J@      I@      R@     �P@      X@     �U@     �Q@     �R@     �T@     �W@     �X@     �[@     �Z@     �X@     �Z@     �X@     �W@     @V@     �Q@     �Q@     @T@      K@     �N@     �L@     �N@     �E@      C@     �A@     �A@     �B@      9@      :@      1@      2@      &@      ,@      "@      &@       @      "@      $@       @      @       @      @      $@      @      @       @      @      �?      @      @      @      �?      @      @      @      @       @       @      �?      �?      @      �?              �?              �?      �?       @              �?              �?              �?               @              �?              �?       @      �?              �?              �?              �?              �?              �?      �?       @      �?       @       @       @      @      @      @      @      @      @      @      @      �?      @      &@      @      "@      @      $@      $@      0@      1@      *@      &@      .@      0@      .@      3@      9@      B@      =@      6@      A@      <@      B@      :@      >@      C@     �D@     �A@      B@     �A@      =@      <@      B@      <@      :@      A@      @@      8@      >@     �A@      >@     �E@     �@@      =@      @@      2@      7@      7@      3@      4@      .@      ,@      0@      ,@      $@      ,@       @      @      "@      @      @      @      @       @      @      @      @       @              @      @       @      �?      �?              �?              �?              �?        �"3      ����	��o���A#*�f

mean squared error��<

	r-squared���>
�L
states*�L	   ����   �@�@    c2A!�'������)�u9}
 A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     p�@     �@     Ȏ@     X�@     H�@     h�@     ��@     h�@     X�@     ؔ@     f�@     T�@     �@     �@     #�@     �@     ��@     ѱ@     ��@     O�@     1�@     ��@     ү@     l�@     ��@     ��@     �@     ��@     Ȳ@     j�@     ��@     U�@     ��@     6�@     {�@     Ⱦ@    ���@     ��@     j�@    �u�@    ���@    �}�@    ���@     ��@     0�@     ��@     ��@    ���@     ��@    ��@     B�@    ���@    �O�@     ��@    �s�@    �s�@    ���@     c�@     ��@    �Y�@    ���@     3�@    �2�@     |�@     ͻ@     ��@     ۷@     y�@     a�@     ��@     �@     �@     P�@     ��@     ��@     
�@     ,�@     ��@     @�@     ^�@     ��@     l�@     ��@     ��@     ��@     "�@     Ȝ@     �@     T�@     �@     4�@     ��@     �@     �@     ��@     ��@     ,�@     ��@     ��@     ��@      �@     А@      �@     4�@     ��@     ��@     (�@     @�@     ��@     ��@     x�@     P�@     h�@     `�@     �@     P�@     (�@     x�@     Ѕ@     H�@      �@     ��@     ��@     ��@     ��@     ��@     ��@     `�@     �~@     ��@     H�@     (�@     ��@     �}@      |@     @     pz@     `{@     �y@     �w@     0x@     �x@     x@     �u@     �x@      v@     �v@     �v@      u@     �u@     0u@     �u@      s@     �q@     �s@     �r@     �w@     w@     0p@     @q@     �q@      o@     �z@      o@     �n@     �m@      o@     �m@     �p@     �i@     �j@     �l@     �j@     �j@      i@     �h@      h@     @e@     `g@     �h@     �d@     �g@     �d@     �d@     `b@      c@     �a@     `d@     @b@     �b@     �a@      c@     �]@      `@     �a@      _@      b@      a@     �^@     �\@      Y@     �X@     �`@     @\@     @V@      Z@     @^@     �[@      Z@     �V@     �Y@     @V@      T@     �U@     �U@      S@     �R@     �T@      U@     @R@      U@      V@     �U@     �Q@     @P@     �Q@     �Q@     �N@      K@     �S@     �K@     @Q@     @Q@      N@     �K@     �N@     �K@     �I@     �J@      K@     �D@      F@     �K@     �P@      I@     �J@     �C@      F@     �L@      E@      L@      C@     �F@     �D@      B@     �B@     �D@      C@      C@      D@      9@      5@      7@      A@      B@      =@      @@      @@     �@@      :@      <@      @@      B@     �C@      A@      6@      7@      5@      3@      6@      5@      9@      4@      1@      >@      7@      .@      (@      2@      *@      5@      2@      0@      2@      &@      2@      4@      0@      0@      .@      $@     ��@      �@      (@      @      $@      $@      ,@      "@      (@      @      @      "@      "@      "@      ,@      .@      "@      @      $@      5@      @      1@      "@      3@      0@      .@      .@      3@      $@      *@      .@      &@      3@      0@      (@      3@      0@      6@      6@      7@      0@      3@      1@      3@      4@      7@      9@      .@      7@      =@      2@      5@      8@      1@      8@      9@      7@      2@      5@      ?@      ?@      :@      6@      6@      :@     �F@      B@      5@      =@      A@      C@     �C@      =@      B@      @@      C@     �E@      A@      C@     �J@      H@      H@     �H@      M@     �J@     �J@     �K@      ?@      G@     �G@     �P@      Q@      I@      J@      J@     �P@      K@      R@      T@      U@     �S@     �P@      T@     @P@     @Q@     �S@     @T@      U@     �W@     �U@     �V@     @T@     �W@     �\@     �Z@     @X@     �W@     �X@     �]@     �[@     �`@     @`@     �\@     �^@     �`@     �_@     @c@      a@     �c@      b@      d@     �c@     @c@     @h@      c@     �g@     �c@     @j@     �g@     `g@     �j@      j@     `i@     @k@     �h@     @i@     �n@     0p@     �m@     `n@     pp@     @q@     Pq@     �p@     pt@     �s@     pt@     �r@     �t@      t@     v@     �t@     �t@     �v@     �w@     w@     �v@     �@     @z@     �y@     @{@      @     p}@     �|@     �{@     �|@      �@      �@     p�@     ȁ@     Ђ@      �@     �@     ��@     ��@     ��@     (�@     І@     ��@     0�@     @�@     ��@     `�@     ��@     �@     ��@     0�@     ��@     8�@     `�@     H�@     ��@     l�@     ؐ@     ��@     ��@     X�@     �@     ��@     p�@     ԓ@     ��@     ��@     ԗ@     p�@     ��@     Ĝ@     ԝ@     �@     ��@     ��@     ��@     ��@     ��@     �@     ̦@     p�@     J�@     ��@      �@     8�@     �@     ��@     ²@     �@     ��@     ��@     k�@     f�@     S�@     �@     �@     ��@    ��@     ��@     ��@    ���@     ��@     ��@    ���@     �@     [�@     �@     W�@    ���@     E�@    ���@     :�@    �"�@     �@    ���@     =�@    �G�@    ���@     ��@    ���@     �@     �@     ׸@     ̶@     ��@     �@     ұ@     �@     �@     ư@     ��@     V�@     b�@     ܯ@     �@     ΰ@     İ@     O�@     ��@     ��@     �@     d�@     Ѽ@     ˶@     P�@     ��@     V�@     ��@     H�@     (�@     @�@     �|@     �z@     �o@     �h@     @v@      J@      5@        
�
predictions*�	   ���   ��6@     ί@!  ��_�"�);�<�YR@2�uo�p�+Se*8俰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&�ji6�9���.����ڋ��vV�R9�})�l a��ߊ4F��f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@{2�.��@!��v�@�������:�              �?              �?       @              �?      @      @      @      @      @      ,@      @@      2@      :@      ?@      C@     �E@      B@     �D@      K@     �M@     �Q@     �Q@     @W@     �V@     @W@      Y@     �Y@     �^@     @[@     �Y@     @Z@     @R@      T@     �U@      U@     �O@     �S@     �J@     �O@     �G@     �G@      F@      @@     �C@      C@      =@      9@      3@      2@      (@      3@      0@      4@      "@      .@      0@      @      $@      @      @      &@      @      @      @      @      @      @      @       @       @              �?      �?      �?      @              @      �?              �?              �?              �?              �?              �?              @      �?              �?              �?      �?               @              �?      �?      �?      @      �?       @              @      @      @      �?      @      �?       @       @      @      @       @      @      @      @       @      ,@      .@      "@      "@      1@      *@      *@      .@      6@      6@      6@      4@      :@     �A@      ?@      =@     �@@      @@      C@     �D@     �B@     �A@     �D@      C@      D@     �F@      D@     �@@      C@     �C@     �B@      H@      7@      A@      ;@      =@      8@      9@      2@      8@      5@      2@      8@      9@      5@       @      ,@      1@      0@      $@      *@      $@       @      @      @      @      @       @      @      @       @      @       @       @       @      �?      �?              @              �?              �?        ����"4      9��	���o���A$*�h

mean squared error���<

	r-squared�X�>
�L
states*�L	    ���   ��@    c2A!f�6
�t��)c���G A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     �@     �@     ��@     ��@     ��@     ��@     0�@     Ј@      �@     ��@     H�@    ���@     Ѵ@     [�@     ��@     r�@     Z�@     S�@     �@     �@     ��@     Ұ@     ��@     B�@     �@     ��@     �@     ��@     ��@     ɲ@     6�@     �@     �@     '�@     ��@     =�@    ���@    �g�@     �@    ���@     M�@     5�@     ��@     ��@    ���@     ��@    �k�@    �,�@     ��@     A�@     ��@    �Z�@     S�@     	�@    �)�@     0�@     ��@     ��@     u�@     E�@    ���@     8�@    ���@     ��@     ��@     Թ@     ׷@     #�@     ��@     ��@     T�@     	�@     ��@     ��@     N�@     
�@     �@     ��@     
�@     �@     ��@     Z�@     ~�@     0�@     R�@     (�@     М@     ̛@     �@     ��@     �@     ��@     �@     �@     �@     H�@     ��@     �@     �@      �@     ؏@     ��@     �@     (�@     Ȏ@     �@     p�@     0�@     ��@     �@     h�@     H�@      �@     <�@     �@     ��@     (�@     ��@     �@     ��@     ��@     p�@     �@     p�@     �@     ��@     ��@     8�@     p|@     �}@     �|@     �~@     �~@     }@     `}@     �@     `|@     �y@     �x@     �y@     `z@      z@     Px@     @x@     @u@     �v@     �w@     @t@     �r@     pt@      s@     Pu@     �s@     �s@     �p@     �q@     @q@      r@     �x@     �{@      r@     �p@     `q@     @l@      p@     0p@     �l@     �k@     `p@     �j@     �i@      l@      k@     �i@     �f@     @h@      f@      j@     �h@     �d@     @f@     @d@     @e@     �e@     �e@     �c@     @d@     `d@     �c@     �a@     �d@     �a@     @`@     �c@     �_@     �`@     �a@     �b@      ^@     @]@     �]@     @_@      \@      _@      Z@     �X@     �X@     �U@      [@     @Y@     �Y@     �V@     �X@     �T@     �P@      Y@     �S@     @Q@      V@     �S@     �T@      S@     �S@     �Q@     �Q@     �R@     �P@      R@      Q@      Q@     �L@     �I@      L@     �F@      O@     �M@      M@      D@     �P@      M@      R@     �E@      ?@     �E@     �F@      E@     �J@     �A@     �L@      L@      B@      I@     �@@     �A@      B@      C@      C@      6@     �A@      @@     �D@      =@      9@     �A@      :@      B@      <@      C@      =@      ;@      <@      8@      5@      :@      <@      ?@      <@     �@@      1@      1@      6@      9@      6@      3@      :@      6@      8@      3@      0@      2@      2@      4@      4@      2@      *@      1@      (@      $@      0@      &@      2@     ��@     8�@      @      $@      @       @      @      $@      @      &@      0@      "@      @      1@      ,@      0@      $@      0@      ,@      (@      (@      2@      0@       @      &@      "@      (@      @      1@      "@      0@      ,@      *@      :@      1@      *@      "@      5@      *@      0@      ,@      8@      8@      <@      0@      5@      1@      2@      3@      6@      5@      8@      7@      ;@      1@      1@      9@      9@     �@@      A@      7@      =@      :@      >@      4@      =@     �@@      A@     �A@      7@     �A@      C@      @@     �A@      F@     �K@     �H@      K@      @@     �F@      I@      L@      L@      F@      E@     �D@      E@     �A@      I@     �I@     @U@     �K@      H@     �K@      Q@     �R@      N@     �P@      Q@     @P@      O@     @V@     �U@      S@      Q@     @S@     @V@     �P@     �R@     @V@     �V@     �[@     �X@     �U@     �[@     �V@     �Y@     �_@     �Z@      `@     �^@     ``@     �`@     �\@     �`@     �a@      e@     �a@     �b@     �c@      f@     �b@     �c@     `f@      g@      f@     �d@     @e@      h@     @i@     �i@     �i@     �j@     `h@      k@     �m@     �i@      n@     `m@     @o@     @o@     �q@     �q@     �n@     `r@     �r@     �r@      t@     �u@     �r@     s@     pt@     �s@     pv@     �v@      w@     @y@     �w@     �z@     �}@     Pz@     `�@     �~@     �|@     �~@     @@     8�@     ��@     ��@     0�@     @�@     ��@     ��@      �@     �@     ��@     �@      �@     @�@     `�@     ��@     ��@      �@     (�@     H�@     �@     ؉@     ��@     ��@     ��@     ��@      �@     �@     ��@     0�@     ��@     ��@     X�@     ��@     ��@     ��@     ��@     �@      �@      �@     ܚ@     x�@     ��@     J�@     .�@     �@     |�@     ʢ@     ƣ@     8�@     $�@     �@     �@     R�@     ��@     ��@     0�@     ��@     ��@     �@     �@     �@     W�@     ��@     X�@     ��@     1�@     �@    �l�@     ��@     W�@    ���@     7�@    ���@     ��@     ��@     ��@     x�@    ���@    �K�@     ��@    ���@     ��@     ��@    ���@    ��@     x�@     �@     ��@     ��@    ���@     &�@     z�@     [�@     S�@     ��@     �@     �@     İ@     ��@     ��@     T�@     �@     +�@     >�@     G�@     ��@     "�@     T�@     ��@     @�@     b�@     ?�@     ��@     ��@     ۷@     ��@     Ы@     �@     \�@     h�@     �@     0@     `u@     x@     �i@     �q@      h@      4@        
�
predictions*�	   @8�   ��?@     ί@!   �>@)5�Z���W@2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���T7����5�i}1�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s���O�ʗ�����Zr[v���u`P+d�>0�6�/n�>O�ʗ��>>�?�s��>>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@{2�.��@!��v�@�������:�              �?              �?      �?              @      �?      @      �?      @      @      @      *@      3@      3@      6@      >@      C@      ?@     �J@      L@      M@      R@      R@     �S@      [@     @V@     �^@     @_@     �[@     �\@     @\@     �V@     @[@     �Z@     @W@     �L@     �P@     �Q@     �N@     �J@      G@      G@      H@      B@     �A@     �C@      5@      >@     �@@      <@      1@      4@      3@      $@      ,@       @      "@      *@      @      &@      @      @      @       @      @       @       @       @      @              @      @       @               @       @      @              �?       @               @      �?      @              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?               @       @       @      �?      �?       @      �?              �?      @              @      @      �?      @      @      @      @      @      @      @      @       @       @       @      @       @      @      $@      2@      (@      &@      .@      ,@      *@      *@      3@      0@      9@      ?@      ;@      9@      <@     �@@      @@      :@      E@     �@@     �A@      A@      @@     �A@     �C@     �B@      A@      ?@     �D@      F@      >@     �@@     �E@     �@@      ;@      3@      0@      ;@      =@      4@      :@      A@      7@      2@      &@      3@      (@      "@      &@      "@       @      @       @      @      @      @      @      @      @       @      @       @      @      @       @               @      �?       @      �?              �?              �?        �v�3      �QIH	�w�o���A%*�f

mean squared errors��<

	r-squared�w�>
�L
states*�L	   ����   ���@    c2A!,W�\�q��)ا"�� A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             l�@     H�@     �@     4�@     ��@     p�@     ؇@     ��@     ��@     ��@     Е@     ĥ@     ��@     ;�@     ��@     ��@     }�@     ?�@     \�@     ��@     c�@     '�@     ��@     د@     ��@     "�@     h�@     �@     y�@     ��@     8�@     ��@     �@     R�@     9�@     Ǽ@    ���@    ���@    ���@     ��@    ���@    ���@     �@     ��@     ��@     n�@    ���@    ���@     {�@     W�@    ���@     1�@    ���@     �@    ���@     ��@    �8�@     I�@    ���@    �%�@     ��@    ���@    ���@    ���@     @�@     ��@     �@     ��@     Q�@     �@     ��@     x�@     �@     ��@     ��@     Ҫ@     ƨ@     ��@     D�@     L�@     ��@     Т@     8�@     F�@     \�@     ܟ@     0�@     p�@     ܛ@     x�@     ��@     ��@     ��@     ��@     h�@     $�@     @�@     ̒@     �@     t�@     �@     ��@     ��@     �@     ��@     �@     P�@     Ў@     (�@     Ј@     x�@     ��@     h�@     ��@     X�@     p�@     P�@     �@     0�@     h�@     h�@     0�@     ��@     �@     ��@     ��@     (�@     ��@     p�@      �@     �@      @      ~@     �|@     8�@     `�@      |@     �}@     �z@     0z@     �{@     v@     @y@      u@     @y@     `w@     Px@     pu@      t@     �u@     �s@     �u@     �u@     �p@     Ps@     Ps@     �r@     `p@     �q@     0}@     @z@      r@     0p@     �o@     �n@     �j@     �l@     �o@     �k@     �i@      i@     �j@     �k@     �k@      k@     �i@     �l@     �i@      e@     �h@     `g@     @i@     �f@      h@     �d@     �g@      h@     �e@     `e@     �b@      a@      a@      b@     �a@      ^@     �^@     `b@      `@     �^@     �[@      ^@     �]@     @[@     �Z@     @\@     �]@     �`@      Z@     �W@     �V@      \@     �V@     �W@     �Y@     �S@     @R@     �Q@     �R@      U@     �T@      T@      O@     �P@      R@     �R@     �S@     @R@     �S@     �Q@     @R@     �M@     �P@     �M@      J@     �J@     �P@     @S@      L@     @P@      I@     @P@      G@      I@     �L@      I@      E@     �D@     �A@      F@      H@      E@      C@      C@      E@     �E@      =@      B@      =@     �A@      E@      ?@      7@     �@@      C@      A@     �F@      =@      A@      >@      8@     �D@      ;@      >@     �A@      3@      ;@      9@      9@      6@     �A@      1@      2@      7@      5@      :@      4@      6@      4@      ,@      6@      3@      3@      0@      1@      ,@      5@      1@      1@      5@      *@      @      0@     @�@     ��@      @      $@      &@       @       @      $@      @       @      "@      ,@      @      (@      4@      "@      &@      "@      *@      (@       @      $@      ,@      0@      "@       @      6@      @      0@      2@      1@      2@      @      0@      0@      8@      4@      .@      2@      2@      4@      4@      1@      .@      5@      :@      9@      9@      1@      8@      6@      4@      :@      9@      5@      4@      @@      8@      4@      <@      :@      6@     �A@      <@      <@      5@      :@      =@      8@      >@     �A@      E@      4@     �C@     �C@      A@     �D@      E@     �B@      G@      H@      J@      E@     �J@      H@      F@     �H@      N@     �D@     �G@      I@     �M@      M@     �N@      O@     �P@     �N@     �K@      M@     @R@      O@      M@     �S@      T@     �R@     �U@     @U@     �S@     �U@     @X@     �X@     �U@     �V@     @U@      T@     @_@      X@      W@     �[@      \@     @^@     @]@     @`@     @^@      `@     @a@     @_@     �a@     `b@      b@      c@      `@     `g@     �e@     �c@     �a@     �e@     �h@     �g@     �h@     `g@      j@     `l@     @i@     @h@      j@     �n@     `l@     �m@     p@      p@      o@     �o@      q@     �q@     �p@     �r@     �r@     �s@     �u@     �t@     �t@     Pt@     �v@     `w@     @w@     �v@     0z@     �z@      y@      {@     �|@     @~@     �~@     �~@     ��@     ��@     ��@     ��@     �@     8�@     x�@     p�@     H�@     h�@     ��@     @�@     h�@     ��@     p�@     0�@     ��@     �@     0�@     (�@     @�@      �@      �@     ؋@     ��@     0�@     h�@      �@     �@     |�@     ��@     ��@     �@     |�@     d�@     ��@     ��@     t�@     \�@     ��@     �@     ��@     Н@     @�@     ��@     ��@     ��@     X�@      �@     ��@     ҧ@     .�@     ��@     ��@     D�@     |�@     ��@     ϱ@     �@     ӳ@     |�@     ��@     ��@     ��@     ˻@     f�@    �F�@    ���@    �|�@     m�@    ���@    �X�@    � �@    �j�@    ��@    �P�@    �Y�@    ��@     G�@     +�@     ��@    ���@    ���@    ���@     
�@    �F�@    ���@     ��@    ���@    �%�@    �q�@     G�@     �@     ��@     J�@     x�@     ��@     ��@     ��@     �@     а@     M�@     b�@     6�@     ��@     ��@     C�@     հ@     ��@     "�@     �@     �@     �@     ��@     �@      �@    �?�@     ��@     ��@      �@     ��@     �@     �@     �t@     �r@      w@     �p@     `x@      =@        
�
predictions*�	   �� �   `ק@     ί@!  (��6@)&S{���U@2�+Se*8�\l�9⿰1%�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��T7����5�i}1���d�r�x?�x��>h�'��I��P=�>��Zr[v�>O�ʗ��>����?f�ʜ�7
?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�              �?       @              �?      @      �?      @       @      @      @      "@      &@      2@      2@      ?@     �G@      9@      A@     �G@      M@     �M@      Q@     @T@      S@     �T@      V@      X@      X@      [@      W@     �Y@     �R@      S@     �R@      Q@     @U@      P@      Q@     @Q@     �E@     �F@      H@      C@     �@@      B@      >@      >@      7@      6@      9@      3@      1@      $@      &@      $@      *@      3@      (@      &@      $@      $@      @      "@      @      @      @      @      @      @      �?      @      @      @       @      @      @       @      �?       @               @      �?              @      �?      �?      �?              �?      �?              �?              �?      �?              �?              �?              �?      @      �?              �?              �?              �?               @      @       @      @              @      @      @       @      @      @      @      @      $@      &@      @      @      *@      "@      "@      $@      &@      .@      .@      6@      ,@      0@      3@      ;@      6@      8@      9@      :@      :@     �A@     �A@     �D@     �B@     �L@      D@     �C@      D@      D@      J@     �I@     �K@      G@     �F@     �H@     �B@     �E@      A@      ?@     �E@      :@     �A@      :@      >@      3@      <@      <@      .@      6@      2@      &@      (@      ,@      &@      .@      @      $@      $@      @      ,@      $@      @      @       @               @       @      �?      �?              �?       @      @               @      �?              �?        ��ʢ3      �gw	��p���A&*�g

mean squared error���<

	r-squared���>
�L
states*�L	   ����    Z�@    c2A!VwQD����)�B뱮� A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ؇@     X�@     ��@     ��@     ȋ@     Ȉ@     @�@     H�@     �@     x�@     (�@    ��@     ��@     ʵ@     ��@     ?�@     �@     ��@     ��@     �@     ��@     ��@     .�@     ��@     R�@     &�@     F�@     ��@     ��@     U�@     ձ@     �@     ��@     ��@     ��@     �@    ���@     e�@     ��@    ��@    ���@    ���@    ���@    �{�@     ��@    �N�@     ~�@     .�@     w�@     ��@    ��@     ��@    ��@     I�@     ��@     ~�@    ���@     d�@    �e�@    �1�@    ���@    ��@    �x�@     �@     �@     ��@     ��@     �@     G�@     6�@     ��@     L�@     >�@     ��@     ^�@      �@     (�@     $�@     ��@     X�@     ��@     ܢ@     ��@     X�@     ��@     ��@     ��@     L�@     2�@     �@     (�@     \�@     ��@     ��@     Ȕ@     �@     `�@     `�@     `�@     �@     \�@     ,�@     �@     �@     �@     ��@     (�@     ��@     �@     0�@     ��@     ��@     ��@     ��@     H�@     (�@     ��@     ��@     p�@     h�@     8�@     ��@     p�@     ��@     ��@     `�@     X�@     h�@     ��@     p@     �}@     p~@     �~@     �|@      ~@     �{@     �z@     Pz@     �x@     py@     Py@     �|@     �x@     �w@     `w@     `u@      y@      u@     �t@     �u@     @s@     �r@     pr@      s@     �q@     `r@     Pr@     @u@     �y@     �s@      y@     �n@     @q@     �n@     �p@     �j@     @l@     �o@      m@     �j@     �k@      j@     @k@      j@     �j@      g@     �g@     �j@     �h@     �e@     �g@     �c@     �e@      h@     �d@     �b@     �c@      c@      e@     �e@      a@      a@      b@     �c@      `@     �_@     @]@     �a@     �^@      ]@     @`@     �`@     @W@     �^@     @^@     �]@      ]@     �T@     �W@     �Y@     @W@     @W@     �W@     �Y@     @T@     �U@     @X@     @V@     �S@     @R@     @Y@     @Q@     �P@     @S@     @Q@      O@     �R@     �K@     �N@     @P@      H@      N@      R@      P@      P@     �L@      P@     �O@      M@      J@     �N@     �D@     �E@      H@      L@      A@     �F@     �H@      G@     �H@     �F@     �A@     �E@      D@      D@     �@@      B@      >@     �B@      A@     �B@      >@     �@@      E@      <@      ;@     �B@     �B@      >@      @@      9@      ?@      <@      8@      7@      @@      9@      @@      7@      3@      0@      7@      ;@      5@      ;@      *@      4@      5@      7@      2@      2@      1@      3@      0@      ,@      (@      3@      2@      (@      4@      .@     @�@     H�@      @      "@      @      @      $@      ,@      *@      &@      @      $@      (@      @      @      ,@      ;@      @      "@       @      *@      3@      ,@      &@      2@      (@      &@      5@       @      9@      (@      0@      "@      &@      .@      0@      3@      6@      5@      $@      7@      6@      ,@      5@      7@      0@      1@      6@      :@      3@      ;@      :@      8@      (@      9@      2@      2@      =@      1@      =@      2@      5@      ;@     �@@      5@      :@     �B@      B@      ?@      A@      C@     �@@     �B@     �A@     �D@      D@     �G@      A@      A@      >@     �I@      =@      C@     �B@     �F@      G@     �D@     �J@      N@     �I@      H@     @P@      P@      S@     �P@     �N@      P@      R@      S@     �R@     �P@     @Q@     @Q@     �J@     @V@     @S@     �W@     �T@      X@     �Z@     �Y@      T@     @[@      [@     @Y@     �W@     �]@     @]@     @Z@      ]@     �]@     �Z@      ^@      _@     �^@      ^@     �b@     ``@     �a@     @c@     �c@      c@     �c@     �c@     �e@     �f@     �g@     �b@     �f@     `g@      i@     `g@      i@      k@      p@     �h@     �m@     �o@     0p@     @p@     �q@     @q@     q@     `o@     0r@     �p@     Pr@     �p@     ps@     �s@     Pt@     t@     �t@     0u@     `y@     8�@     P{@     @x@     0z@     0{@     `|@     �y@     �|@     �~@     Ȁ@      �@     �@     �@     �@     �@     �@     �@     `�@     ��@     �@     ��@     �@     ��@     Ȅ@     0�@     X�@     p�@     ��@      �@     (�@     ��@     ȋ@     Њ@     H�@     0�@     p�@     ��@     �@     H�@     @�@     (�@     ܕ@     ��@     ��@     ��@     P�@     @�@     ��@     4�@     �@     ��@     �@     �@     ԝ@     ��@     �@     ��@     ��@     V�@     ʦ@     ��@     �@     0�@     ��@     ޮ@     ��@     �@     Ͳ@     "�@     ��@     C�@     ��@     �@     &�@     ļ@     D�@     ��@     �@    ���@     ,�@    ���@     ?�@    ���@     x�@     *�@    �s�@    �[�@     ��@     -�@    ���@     �@     ��@    �#�@    ��@     ��@     ��@     ��@    �)�@     Y�@    ���@    ���@     M�@     $�@     `�@     ��@     4�@     r�@     ��@     �@     Ѱ@     ��@     ��@     ��@     \�@     R�@     -�@     &�@     İ@      �@     �@     D�@     s�@     ��@     ~�@     C�@     �@    �/�@     b�@     <�@     ��@     ȅ@     ��@     0@     �v@     0q@     x@     @s@     �{@      ;@        
�
predictions*�	   @���   @Q@     ί@!  �o�1@)�x�W7�Y@2�+Se*8�\l�9���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9��5�i}1���d�r�x?�x���h���`�>�ߊ4F��>pz�w�7�>I��P=�>��Zr[v�>�T7��?�vV�R9?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@!��v�@زv�5f@�������:�              �?               @      @              @      @      �?      @       @      0@      1@     �A@     �@@     �D@      I@      K@      F@      O@     �N@     @Q@      T@     �Q@     �X@      R@      W@      X@      W@     @R@     �R@     �Q@      R@     �P@     �S@      T@     �P@     �L@      J@      F@     �C@     �@@      A@     �@@      6@      A@      2@      9@      8@      2@      2@      1@      0@      *@      &@      "@      &@       @       @      @      @      @      @      @      @       @      @      @      @      �?       @      @              �?      @      �?      @      �?       @      �?      �?              �?               @              �?              �?              �?      �?              �?              �?      �?              �?               @               @       @      �?       @       @       @      �?      �?      �?      @      �?              �?       @      �?              �?      @      @      @      @       @      @      @      @      $@       @      "@       @      ,@      $@      ,@      .@      *@       @      5@      5@      &@      5@      ?@      7@      =@      <@      8@      4@      B@     �I@      A@      H@     �K@     �J@      O@     �N@     �M@     �I@     @Q@     �N@      M@     �L@     �D@      F@      E@      B@     �B@      A@      6@      ;@      1@      ;@      ;@      7@      4@      6@       @      5@      4@      "@      ,@      $@      "@      &@      @      $@      @      @      @      "@      @       @      �?       @      @      @       @      @               @      @              �?      �?              �?        �a�B3      �%��	�3p���A'*�f

mean squared errorB@�<

	r-squared(9�>
�L
states*�L	   ����   �x�@    c2A!G��I��);�))�{A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     �@     $�@     8�@     0�@     ��@     ��@     `�@     ��@     ��@     �@     ��@     �@     ?�@     ָ@     ��@     #�@     ��@     <�@     ��@     ȯ@     ��@     ��@     !�@     ��@     R�@     "�@     @�@     �@     ��@     ʲ@     �@     �@     ��@     ��@     �@     q�@     ��@     ��@     ��@    �
�@    �W�@    ���@     ��@     3�@     ��@     ��@     "�@    �8�@     T�@     ��@    �D�@     ��@     ��@    ���@     ��@     �@    ��@    �x�@    �_�@     ��@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     C�@      �@     .�@     ��@     ��@     Ԩ@     .�@     b�@     ��@     �@     D�@     �@     >�@     \�@     �@      �@      �@     \�@     \�@     �@     D�@     h�@     �@      �@     ��@     ��@     ��@      �@     ��@     Ȑ@     �@     ��@     \�@     d�@     ��@     ��@     ��@     `�@      �@     �@     ��@     ��@      �@     h�@     ��@     @�@     ��@     H�@     h�@     ��@     �@     0�@     ��@     8�@     ��@     �@     (�@     ��@     �@      @     �@     �~@     �~@     �}@     |@     �z@     px@     `{@      z@     �x@     �v@     �v@      w@     �w@     `y@      w@     �u@     �w@     �r@     0t@     @q@     �s@     �q@     �t@     �p@     �q@     �y@     �t@     �o@      o@     �z@     �p@     `l@     �k@      m@      l@     �l@     �k@     `i@     �g@     �j@     �i@     `j@      h@     �f@      h@     �g@     �g@     �c@     �f@     �e@      f@     �e@     �b@     @f@      c@      c@      e@      c@      a@      c@     @c@     @`@      `@     �`@      ]@      ^@      ]@     @^@     �^@     @\@     @[@      [@     �]@     �X@      Y@     @W@      U@     @Y@     @X@     �W@      W@     �Z@     �U@     @W@     @S@     @S@     �U@     @Q@     @Y@     �Q@     �R@      N@     �R@      Q@     �K@      R@     @U@     �O@     @R@      F@     �R@     �Q@     �J@      G@      N@      N@     @Q@     �L@     �E@      C@      G@      J@     �G@     �B@      H@     �@@     �A@      I@     �C@     �C@      <@     �F@     �G@      D@      >@      B@      A@     �C@     �H@      @@      ?@      B@      8@      B@      E@      >@      6@      <@      :@      6@      6@      9@      *@      ?@      7@      7@      5@      7@      7@      3@      :@      5@      .@      6@      4@      3@      8@      7@      0@      :@      0@      ,@      (@      0@      (@      5@      "@      *@      3@      4@     ��@     ��@      @      $@      @      @      &@      $@      "@      ,@      .@      ,@      &@      1@      (@      @      (@      ,@      0@      @      "@      @      "@      (@      $@      *@      &@      0@      .@      (@      1@      .@      (@      &@      0@      4@      &@      5@      5@      9@      2@      1@      3@      5@      2@      .@      7@      .@      6@      7@      4@      9@      0@      7@      5@     �@@      5@      ;@      ;@      3@      @@      9@      <@      >@      7@      <@      @@      ?@      @@      <@     �C@      A@     �A@      @@      I@     �E@      C@      E@      B@     �H@      F@     �H@     �C@      G@      E@     �E@     �K@     �M@     �J@      N@      J@     �P@     �K@      M@      N@     �O@     @P@     @P@     �Q@     �P@     �Q@      S@     @Q@     �M@     �S@     �Q@     �T@      Y@     �W@     �S@      W@     @W@     �Y@      \@     �W@     @\@     �Y@     @Z@      ^@      [@     �`@     @^@     �[@     @[@     �`@      a@      c@     �`@     �`@      b@     �b@     �c@     �e@     �b@      a@     �b@     @f@     `g@      e@      h@      e@      i@     �i@     �g@     @h@     �k@     �i@      j@     pq@      o@     pp@     Pp@     �p@     �q@     p@     ps@     �z@     �r@     0t@     �s@     0t@     �t@     u@     Pu@     w@     @u@     0v@     Py@      y@      {@      z@     |@     P|@     �}@     p@     @     h�@     `�@     ��@     `�@     ��@     �@     ��@      �@     ؄@     x�@     ��@     �@     0�@     p�@     ��@     �@     `�@     �@     P�@     (�@     h�@     h�@     x�@     h�@     �@     �@     ؏@     �@     ��@     ��@     ȑ@     p�@     ��@     �@     �@     P�@     ��@     �@     ��@     ��@      �@     H�@     \�@     T�@     H�@     ��@     ��@     ��@     r�@     T�@     �@     v�@     �@     ��@     ��@     ή@     �@     ��@     K�@     ��@     �@     ӷ@     !�@     ��@     l�@     :�@    ���@    �+�@    ���@     ��@     ��@    ���@     ��@    ���@     5�@    ��@     ��@     �@     �@    ���@     ��@    ���@     ��@     ��@    �9�@     d�@     ��@    ���@     ��@     (�@     ��@     �@     ��@     e�@     Ҵ@     ��@     6�@     װ@     ��@     k�@     A�@     �@     L�@     `�@     �@     ܰ@     �@     ��@     ��@     W�@     Ӵ@     ��@     һ@     J�@     ��@     �@     d�@     ��@     ��@     @�@     �@     0�@     X�@     Pu@     Pq@     P|@     0�@     �T@        
�
predictions*�	   ��!�   �	]@     ί@!  @��s	@)���'W@2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !��.����ڋ��vV�R9��5�i}1���d�r�>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��82?�u�w74?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�              �?      �?      �?      @      �?      @      @      @      $@      (@      &@      0@      A@      B@     �E@     �C@      H@      J@      M@     �R@     @X@     @X@     @X@     �[@     �Y@     �[@     �[@     @]@     �Z@     @W@      Z@      U@     @S@     @S@      Q@      P@     �P@     �N@      M@      F@      I@      G@     �@@      ?@     �A@      8@      :@      7@      7@      1@      (@      4@      6@      *@       @      "@      0@      @      (@      @       @      @      �?       @       @      @       @      @      �?      @       @      @              �?      �?      �?      �?      @      @      �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?               @      �?               @               @              @      �?       @       @      @       @      @      @      @       @              @      @       @       @      @      (@      "@       @      @      @      $@      .@      1@      &@      "@      3@      &@      0@      4@      3@      .@     �A@      5@      4@      7@     �@@      8@      D@      ?@      @@      <@     �A@     �@@      ?@     �F@      D@      A@      =@     �A@     �@@      A@      ;@     �@@      @@     �B@      >@      6@      6@      6@      6@      0@      *@      <@      .@      5@      *@       @      *@      ,@      "@      &@      @      $@       @      @      @      @       @      @       @      @       @       @      �?       @      �?       @       @              �?              �?        )R_��3      _  	�KQp���A(*�g

mean squared error��<

	r-squared�B�>
�L
states*�L	   ����   �-�@    c2A!3�㍀��)C�� /A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ̕@     ��@     p�@     p�@     P�@     Ї@     ��@     `�@     P�@     H�@     ��@     d�@    ���@     Ӵ@     ��@     �@     մ@     ��@     �@     ��@     G�@     ��@     а@     �@     *�@     ��@     ��@     ��@     -�@     �@     ��@     a�@     ��@     ��@     
�@     ��@     W�@     '�@     ��@    ���@     ��@     ��@     ��@    ���@    �;�@    �o�@    �z�@    ���@     V�@     �@    �o�@     8�@    ���@     ��@    ���@     ��@    ��@     k�@     .�@    ���@    ���@     ��@     %�@    ���@     ܿ@     ߽@     ��@     ��@     2�@     �@     �@     A�@     ��@     �@     2�@     j�@     �@     �@     ��@     �@     Ф@     l�@     l�@     ��@     �@     �@     Ğ@     Л@     ��@     d�@     ��@     ��@     ��@     ̖@     ��@      �@     �@     �@     4�@     ��@     \�@     ��@     ԑ@     ��@     ,�@     �@     ��@     ��@     �@     H�@     �@     ��@     @�@     ��@     ��@     ��@     ��@     h�@     8�@     ��@     ��@     �@      �@     ��@     �@     ��@     X�@     �@     ؁@     X�@     P@     �~@     @{@     �|@      }@     @{@     0z@     �x@     `y@     �x@     �x@      x@      v@     �v@      w@     �t@     �v@     �v@     @u@      v@     �t@     �r@     �r@      s@     pr@     �q@     �q@      v@     w@     p@     r@     `t@     @v@     `n@     `n@     �j@     �n@     �f@      m@     �k@     �i@     �i@     �k@     @j@      k@     �g@      f@     �f@     `g@      h@     �f@      d@     @e@      h@     `f@     @f@      c@     �b@      d@      c@     �c@     �b@     @c@     �a@     �a@     �`@     �b@     �]@     �W@      `@     �^@     �W@      _@     @_@      ]@     �X@     @Z@      V@     �[@      U@     @Y@      W@      Y@     �S@     �R@      V@     �R@     �X@      U@     @R@     �T@     �N@     @R@     @S@     �Q@     �P@     @S@     @S@      L@     �K@     �P@     �I@      I@     �P@     �P@     �P@      P@     �C@     �L@     �G@     �H@     �J@     �I@      H@     �D@      O@      G@     �G@     �B@     �C@      G@      @@      @@     �A@      F@     �A@      B@     �D@      D@     �C@      C@     �@@      ?@      C@      =@     �D@      =@      B@      B@     �A@      7@      ;@      9@      7@      5@      <@      ?@      6@      4@      6@      0@      2@      6@      0@      0@      8@      (@      8@      6@      *@      1@      6@      ,@      5@      .@      ,@      6@      1@      "@      0@      (@      (@     �@     ��@      @      (@      @      *@      $@      $@      *@       @      &@      *@      @      .@      $@      0@      0@      @      &@      (@      0@      (@      *@      &@      ,@      2@      ,@      ,@      .@      *@      &@      .@      2@      (@      .@      (@      0@      2@      7@      7@      &@      4@      ?@      3@      &@      2@      1@      7@      8@      >@      :@      *@      <@      1@      6@      1@      0@      B@      :@      4@      <@      9@      ?@      D@     �C@      D@      @@      =@      ?@      E@     �A@      A@      C@     �D@      F@     �D@      J@     �J@      F@     �C@      E@      A@     �J@     �C@      J@     �G@      O@      G@     @Q@      Q@     �L@     �J@     �O@      J@     �S@      L@      O@     �P@      Q@     �Q@     @S@     @Q@     @T@     @T@     @P@      Q@     �Y@     @V@      V@      U@      W@     @T@     @Z@     �T@     �Z@     �W@      [@     �Z@     �Y@      V@      ^@     �]@     �`@     @^@     `a@     �`@     `a@     �_@      `@      c@     �a@     �e@      b@     �`@      d@      e@     @e@     @f@     �e@     `f@     �h@     �i@     `g@     @g@     `m@     �l@      h@     �n@     �l@     `n@     �x@      q@      m@      q@     �o@     �q@     @q@     Pp@     0r@      t@     �u@     ps@     `w@     �t@     @u@     �v@     �u@     �w@     `y@     y@      |@      z@     �|@     �~@     ~@     �~@     Ѐ@     ��@     ��@     H�@     ��@     ��@     ��@     p�@     x�@     ��@      �@     �@     h�@     ��@     �@     ��@     ��@     І@     �@     ��@     ؊@     Ћ@     �@     ��@     �@     0�@     h�@     ��@     ��@     �@     ��@     `�@     �@     8�@     <�@     �@     @�@     x�@     ��@     �@     \�@     X�@     ��@     ��@     �@     �@     8�@     ʥ@     L�@     f�@     ��@     D�@     h�@     `�@     �@     �@     #�@     	�@     w�@     ��@     s�@     �@     ��@     ��@    ��@    �3�@    ���@    �
�@    �_�@     "�@     �@    ��@     ��@    ���@     ��@    ���@    ��@    ��@     ��@    ���@     ��@    �)�@     ��@    ���@     -�@    �#�@     ��@     ��@    �#�@    ���@     �@     V�@     o�@     �@     �@     �@     �@     ұ@     W�@     V�@     ȱ@     8�@     �@     d�@     ��@     �@     P�@     ��@     �@     ��@     ͵@     �@     g�@     ��@     (�@    ��@     Ԫ@     �@     t�@     `�@     0�@      �@     �@     �x@     �p@     �v@     ��@      t@        
�
predictions*�	    p\�   ��X@     ί@!  �wE[ @)�Vѫ9U@2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�a�Ϭ(�>8K�ߝ�>1��a˲?6�]��?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?{2�.��@!��v�@�������:�              �?      �?      �?              �?       @      @       @      �?       @      "@      ,@      (@      5@      6@      <@      =@      >@     �E@      F@      F@      F@      I@     �P@     �Q@     �R@     �T@     �S@      U@     �R@     �V@     �R@      W@      U@     �V@      R@      H@      N@     �N@     @Q@      L@      I@      D@     �A@      C@     �B@     �A@      C@      7@      1@      9@      3@      *@      .@      4@      (@      6@      .@      0@      *@      ,@      @      @      @      @      (@      @      @      @      @      �?      @      @      @      �?      @       @      @      @       @       @      @      �?      @      �?              �?      �?              �?       @       @              �?      �?              �?              �?              �?              �?      �?      �?              �?      �?      �?               @      �?               @      @              @      @      @      �?      �?      �?      �?      �?      @      @      @      @      $@      @      $@       @      $@      1@      @      2@      1@      5@      1@      0@      6@      ?@      8@      =@      C@      C@     �C@      C@      C@     �K@      G@     �L@     �E@      J@     �D@     �I@      F@      J@      G@     �K@      G@     �E@      G@      K@      D@     �B@      D@      =@      <@      4@      :@      7@      3@      3@      3@      $@      ,@      *@      &@      *@      .@      .@      "@      $@       @      @      @      @      @      @       @      @      @      @       @      @      �?      �?       @      �?              @      �?              �?        tv��b4      8�W	�jqp���A)*�h

mean squared error��<

	r-squared �?
�L
states*�L	    ���   @k�@    c2A!���n��)�`"�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     �@     �@     ��@      �@     (�@     p�@     Ȇ@     `�@     �@     H�@     4�@    ���@     �@     ��@     �@     8�@     ��@     4�@     Y�@     ��@     /�@     "�@     V�@     ��@     8�@     �@     ��@     �@     ��@     Ѳ@     ȴ@     �@     6�@     ��@      �@     �@     T�@    �{�@    �>�@    ���@    �;�@    ���@     C�@    ���@    ���@     '�@    ���@     %�@     �@     ��@    ���@    ���@    �x�@    ���@    ���@     ��@    �m�@     ��@    ���@     ��@     ��@     ��@     �@     $�@     ��@     �@     ��@     -�@     γ@     .�@     J�@     "�@     n�@     "�@     ̪@     ��@     \�@     �@     �@     ��@     ��@     �@     v�@     L�@     ��@     ��@     �@     ؙ@     h�@     h�@     ��@     ��@     ��@     ��@     �@     ��@      �@     ��@     4�@     (�@      �@     ��@     ��@     ��@     �@     ��@      �@     `�@      �@     ��@      �@     ��@     ��@     �@     �@     ��@     ��@     ��@     ��@     Ј@     �@     8�@     �@     H�@     X�@     ؁@     P�@     ؁@     P�@     �@     �|@     @}@     �|@     �}@     p{@     0{@     �|@     `{@     `z@     �w@     v@     0x@     �v@     0w@     �w@     x@     �v@     �u@     pt@     0u@     pq@     pr@     �s@     �p@     �p@     s@     q@     `z@     @p@      q@     @p@     �w@     @o@      m@     �o@     �n@     �l@     �i@     �k@     `i@      j@     `g@     �j@      h@      j@      i@     @f@      h@     �h@      i@     @f@     �h@      b@     �b@      h@     �d@      c@     @c@     `e@     �_@      c@     @\@     �c@     @`@     `a@     �`@     @a@     �]@     �]@     �Y@     �^@      [@      Z@     �\@     �Z@     @X@     �`@      W@      \@     @V@      W@     �W@      W@     �T@      W@     @R@     �Q@     �S@     @S@      R@     �U@     �O@     �S@      R@      P@      N@     @Q@     @P@     �T@     @P@     �P@     �R@     �P@     �J@      K@      I@      P@      K@      M@     �K@     �M@     �H@      E@      J@     �G@      D@      C@     �F@      K@     �M@      ?@     �M@      F@      J@      @@      ?@      F@      G@      D@      C@      A@     �A@     �D@      A@     �B@     �A@      <@      ;@      >@      6@      9@      <@      @@      :@      <@      ;@      5@      7@      =@      ;@      3@      5@      5@      7@      7@      :@      1@      3@      1@      @      2@      2@      5@      4@      ,@      .@      *@      0@      &@      $@      2@     ��@     ȁ@      $@      "@      *@      @      @       @       @      &@       @      @      "@      &@      *@      @      *@      $@      $@      1@      *@      @      $@      1@       @      4@      .@      .@      (@      1@      4@      2@      "@      &@      2@      (@      &@      4@      2@      8@      *@      0@      4@      2@      :@      7@      ;@      5@      6@      .@      ,@      >@      >@      1@      &@      3@     �@@      8@     �B@      :@      >@      6@      8@      B@     �@@      <@      @@      B@     �@@     �A@      D@      C@      A@      E@      G@      H@     �K@     �C@     �H@     �D@      B@      I@     �G@      F@     �E@     �I@     �E@      I@      G@      L@      J@     �P@     �M@     �M@     �M@      L@     @Q@     �S@     �N@      S@     @R@      V@     �Q@     �S@      S@     �Q@     �T@     @U@     @U@     @T@     @S@     �V@      X@     �Y@      Y@     �[@      T@      W@     @X@     �[@     �V@      `@      X@     @[@     @`@      `@     �^@     �_@      c@     �a@     `c@     @a@     @b@     �a@      f@      d@     �g@     �d@      e@     �g@     `f@     �g@     �c@     �j@      h@      j@      n@     �j@      x@     �l@     �h@     @o@     �n@      p@     @r@     �n@     �q@     �r@     �q@     ps@     �p@     �t@     w@     0u@     �u@     0w@     �v@     py@     z@      z@     �z@     P~@     �{@     �}@     p}@     �}@     0�@     P}@     x�@     ��@     x�@     (�@     `�@     �@     ��@     ��@     x�@      �@     p�@     X�@     ؆@     ��@     ��@     ��@     h�@     ��@     ��@     ��@     ��@     ��@     ��@     h�@      �@     �@     ,�@     ��@      �@     4�@     @�@     X�@     Ȕ@     �@     �@      �@     @�@     ̚@     ��@     @�@     �@     `�@     ¡@     ��@     ��@     £@     b�@     2�@     ��@     �@     8�@     ��@     �@     0�@     ��@     m�@     �@     ��@     .�@     m�@     i�@     Q�@     2�@     '�@     H�@     ��@     m�@     ��@    ���@     P�@     y�@     ��@     ^�@     ��@     ��@    ��@     �@    �;�@     =�@    �@�@     ��@    ���@     ��@    ���@     ��@     �@     k�@    �7�@     �@     $�@     ��@     U�@     i�@     [�@     k�@     �@     Ʊ@     �@     p�@     ү@     D�@     �@     ��@     ΰ@     \�@     в@     �@     ��@     ��@     M�@     ��@     ��@     ��@    ��@     ��@     ��@     ��@     0�@     ��@     �}@     `�@     `v@     �q@     �u@     x�@     �x@        
�
predictions*�	   �k��    �@     ί@!   �2�)h�(^
X@2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��T7����5�i}1�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���ߊ4F��h���`�8K�ߝ뾞[�=�k���*��ڽ��uE����>�f����>pz�w�7�>I��P=�>����?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�6v��@h�5�@{2�.��@!��v�@�������:�              �?      �?      @              @      @      *@      @      ,@      ;@      5@     �B@      =@      F@      D@     �G@     �H@      P@     �R@      P@      P@     �Q@     �R@     @S@      P@      U@     @P@      W@     @Q@     �Q@      Q@     �R@     �S@     �P@     �P@     �H@     �L@     �C@      D@      I@      D@     �H@      >@      @@      A@      =@      5@      7@      9@      .@      4@      .@      ,@      2@      .@      ,@       @      @      @      @      @      @      @      @      @      @      @      @      @      �?       @       @       @       @      @              �?               @      �?              �?              �?      �?              �?               @               @       @      �?      �?              �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?      �?      �?               @              �?              �?               @      @      �?       @              �?      @      @      @      @      @      @      @      �?      @       @      "@      "@      "@      ,@      .@      *@      ,@      .@      ,@      :@      (@      =@      9@      9@     �@@      @@     �E@      @@      J@      C@     �D@     �C@      K@     �F@     �H@      I@      C@      G@     �D@     �J@     �J@     �H@      D@     �H@     �D@     �B@      B@     �A@     �B@      A@      <@      8@      :@      4@      3@      3@      7@      0@      4@      1@       @      $@      *@       @      @       @      @      @      @      @      @      @      @      @       @      �?      @               @      �?      �?      �?              �?              �?        �=�J�3      �=%�	���p���A**�g

mean squared erroroh�<

	r-squaredt�?
�L
states*�L	   `t��    ��@    c2A!wP��4���)�H,��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             x�@     ��@     ��@     Џ@     ��@     0�@     P�@     ��@     ��@     ��@     h�@     ԣ@    ���@     ��@     e�@     ��@     \�@     ��@     �@     ��@     ��@     �@     k�@     ��@     ^�@     �@     f�@     �@     �@     Ӱ@     W�@     �@     G�@     ��@     ��@     ��@     s�@     ��@     G�@     ��@     k�@     ��@     �@     ��@    ���@     ��@     ��@     ��@     �@     ��@     Q�@     B�@    ���@    ���@    ��@     !�@     ��@    ��@    ���@     �@    ���@    ���@    �h�@    ���@     ۾@     ʼ@     ��@     ��@     ��@     y�@     h�@     Ʊ@     (�@     �@     |�@     0�@     :�@     ��@     ��@     l�@     V�@     ��@     `�@     �@     :�@     :�@     �@     L�@     D�@     �@     t�@     ��@     ��@     Ė@     �@     ��@     l�@     �@     ��@     �@     D�@     ��@     Ԑ@     ��@     H�@     �@     <�@     ��@     Ѝ@     ��@     �@     ��@     X�@     Ј@     ��@     ��@     ��@     ��@     ��@     ��@     h�@     `�@     �@     p�@     Ђ@     ��@     X�@     �@     ��@     Ȃ@     8�@     @|@     |@     �|@     �{@     `|@     �y@     z@     �y@     �x@     v@     @y@      w@     �w@     0w@     �v@     @x@     �u@     �t@     @u@     �t@     �s@      u@     �q@     0s@     Pq@      q@     p@     �y@     �q@     �q@      n@     �u@     @u@     @o@      k@      l@     @m@      n@     @g@     �i@     �h@      o@     �h@     �g@     �h@      j@      i@      h@     �e@     �e@     �e@     �f@      k@     �f@     �g@     �a@     �e@     �c@     �d@      f@     �a@     �a@      `@     �_@     �`@     @`@     �_@     �`@     �_@     �_@     @Z@      `@     @]@     @Z@     �Z@      Y@     �]@      Z@      Z@     @V@     �U@     �R@     �T@      T@     �V@     �U@      U@     @W@     @T@      Q@     @R@      Q@     @Q@     �Q@     �Q@     �L@     @R@      N@     �Q@     �N@     �M@     �Q@     �Q@     �S@     �Q@     �G@     �S@     �O@     �G@      O@      L@      F@      P@     �J@     �K@      I@     �M@     �K@     �F@      H@     �H@      F@      E@     �E@      D@     �C@      E@      D@      F@      C@     �E@      @@      =@     �A@      ?@      C@     �@@      3@      A@      <@      <@      C@      7@      2@      9@      7@      8@      :@      8@      1@      9@      3@      $@      :@      5@      7@      ;@      0@      4@      3@      1@      8@      1@      3@      1@      >@      4@      .@       @      $@      3@     ��@     ȁ@      $@      (@      "@      &@      @      "@      @      0@      @      (@      $@      (@       @      *@      $@       @      ,@      (@      4@      @      "@      6@      &@      (@      (@      6@      *@      0@      1@      *@       @      "@      *@      ,@      &@      &@      .@      5@      3@      .@      ;@      ;@      9@      9@      9@      (@      ;@      1@      7@      7@      <@      7@      <@      6@      7@      ;@      7@      5@      2@      :@      =@      >@      =@      B@      <@      C@      ;@      >@      A@      C@     �F@     �B@      A@     �D@     �E@      >@      H@      B@      K@     �C@      D@     �C@     @P@      H@      G@      I@      P@     �K@      M@     �G@      L@     �Q@     �I@     �J@     �P@     �N@     �O@     �Q@      O@     �R@     �P@     @P@      Q@     �S@     @Q@     �Q@     @T@     @V@     �Y@      X@     �R@     @X@     @U@     �Z@     @Y@     �Y@      ]@     �\@     @W@      \@      ^@     �X@      _@      `@     �]@     �`@     �_@     �a@     �`@     �`@     �_@     `f@      c@     �c@     �f@     �a@     `f@     �d@      f@     �e@     `e@      k@     �k@     �l@     �t@      i@     �h@      k@     @q@      o@     `n@     Pp@     �n@     0r@      r@     `q@     �p@      s@     r@     0s@     �s@     @s@     Pw@     �x@     0x@     x@     @x@     @x@     �{@     �y@     �}@     �|@     �}@     �~@     ��@     @�@     �~@     �~@     �@     h�@      �@     �@     ��@     ��@     ��@     x�@      �@     ��@     ��@     �@     x�@     ��@     �@     0�@     ��@     H�@     ��@     ؎@     ��@     H�@     �@     ��@     ��@     D�@     x�@     �@     ��@     ��@     ��@     ��@     �@     h�@     ��@     Ț@     4�@     ��@     ��@     ��@     $�@     ��@     Ԥ@     ��@     &�@     $�@     $�@     |�@     ��@     �@     �@     s�@     ��@     R�@     ,�@     ��@     �@     ��@     
�@     i�@     ��@     �@     R�@     ��@     A�@     ��@    ���@     -�@     ��@    ���@     ��@    ���@     ��@    �@�@    ���@    �C�@     ��@    ���@    ���@     ��@    �F�@    �$�@    ���@    ���@    �!�@    ���@     ��@     ��@     ۹@     �@     u�@     d�@     �@     ��@     �@     �@     �@     G�@     ��@     ��@     |�@     @�@     ��@     ֱ@     ]�@     ��@     ��@     @�@     ��@     `�@     ��@     ��@     *�@     �@     X�@      �@     @~@     �w@     �{@      u@     �q@     `u@     0�@     @w@        
�
predictions*�	   ����    ;~@     ί@!  ���@@)P~}�1X@2�+Se*8�\l�9⿰1%��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
�I��P=��pz�w�7���f����>��(���>����?f�ʜ�7
?>h�'�?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�              �?      �?              �?      @       @      $@      @      *@      $@      2@      .@      6@     �@@      <@      @@     �D@     �J@     @P@     �K@     �P@      W@     �Y@     @W@      R@     �Z@     @Y@      Z@      Z@     @W@     �U@     �T@     �R@     @S@      O@     @P@     �E@     �@@     �F@     �F@      @@      ?@      ?@      >@      ;@      4@      <@      0@      (@      *@      *@      *@      ,@      &@      1@      $@       @      &@       @      "@      @      @      @      @      @      @      @      @      @      @      @      �?      �?       @      �?       @      �?      @      �?      �?              �?              @               @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      @      �?      �?       @      �?      �?      @      @              @      �?       @      @       @      @      �?      @       @       @      @      @      @      @       @      "@      @      $@      &@      &@      "@      &@      1@      "@      *@      .@      .@      (@      :@      @@      7@      <@      C@      C@      ?@      H@      B@      F@      D@      C@     �D@      K@     �K@     �H@     �@@      D@     �D@      E@      H@     �C@      F@     �K@     �@@      B@      >@     �C@      >@     �@@     �@@      7@      9@      7@      .@      >@      2@      $@      *@      .@      &@      $@      *@      @      @      "@      @      @      @      @       @      @       @      @      @      @               @       @       @              �?              �?        ����b3      ��v	���p���A+*�f

mean squared error�θ<

	r-squared�b?
�L
states*�L	    ���   ���@    c2A!�~��tv��)��QpA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     0�@     ��@     ��@     ؆@     0�@     ��@     ��@     �@     �@     �@    �v�@     ��@     �@     �@     ?�@     ��@     @�@     u�@     ��@     �@     ��@     ��@     l�@     ��@     B�@     خ@     Ű@     �@     ��@     <�@     �@     t�@     |�@     H�@     ��@    �r�@    ���@     A�@    ��@    ���@     ��@     ?�@     |�@     ��@    �[�@    �=�@    ���@    ���@    �E�@    �J�@    ���@     ��@    ���@    ��@     ��@    � �@    ���@     [�@    ���@     ��@     ��@    �+�@     �@     n�@     &�@     _�@     C�@     $�@     ;�@     ��@     x�@     ��@     �@     ��@     ��@     ��@     ��@     p�@     z�@     n�@     ��@     ,�@     p�@     ��@     L�@     �@     ��@     �@     L�@     ��@     `�@     ��@     �@     P�@     ��@     `�@     $�@     ܒ@     ؑ@     ��@     �@     �@     ��@     h�@     ��@     ��@     4�@     ��@     ��@     `�@     �@     ��@     x�@     Ї@     ��@     @�@     0�@     0�@     ��@     ��@     `�@     `�@     ��@     ��@     ��@     h�@     ��@     h�@     @     �}@     P~@     |@     �}@     |@     �{@     �v@      {@     0{@     0{@     x@     �w@     �v@     Pw@     �v@     �v@     �v@     �u@     @u@     s@     @t@     �r@     �s@     �r@     0t@     �q@     @q@     �z@     �p@     `q@     @o@     �n@     �r@     @u@     �o@     @n@     @k@      n@      o@      l@     �j@     �n@     `j@      g@     �e@     �i@     �e@      h@     �f@     @h@     @g@      f@      f@     �f@     �d@     @h@      e@     �a@     �c@     �`@      c@     �b@     @a@      c@     �b@     @_@     @`@      `@     @^@      `@     �]@      ^@     �Z@     �W@     �W@      Y@      \@     �V@     �_@     �Z@      X@     �X@     �W@     @R@      W@      Z@      S@     @U@      T@     �T@     �X@     �R@     �T@     @P@     �N@      K@     @R@      S@      Q@      S@     �L@     �N@     �O@     @Q@     �R@      M@     �K@      O@      H@      O@     �I@      H@      N@     @P@      M@      G@      N@      P@      F@      F@      L@     �H@     �B@      J@     �C@      H@     �C@     �H@      @@      C@      ;@      5@      @@     �A@      I@      :@      9@      4@      8@      <@     �@@      ;@      9@      8@      =@      ?@      4@      9@      6@      =@      >@      6@      <@      >@      ,@      3@      3@      9@      7@      ,@      5@      ,@      5@      4@      *@      5@      @      (@      *@      ,@      (@     Ѝ@     ��@       @      $@      "@      (@      @      "@      ,@      (@      ,@      $@      $@       @      @      .@      *@      $@      .@      ,@      ,@      &@      $@      "@      *@      .@      2@      (@      *@      ,@      $@      .@      @      2@      *@      .@      2@      ,@      0@      8@      3@      7@      :@      6@      @@      1@      3@      *@      $@      0@      >@      ?@      .@      7@      ?@      8@      6@      A@      9@      2@      ?@      ?@      3@      6@      >@      7@     �@@      @@     �C@      C@      A@     �C@      C@      F@      ;@      I@      K@     �B@     �@@      F@      G@      H@      E@      K@      F@     �J@     �B@      C@     �H@     �L@      K@     �F@      M@      R@     �P@      Q@     �N@     �M@      O@     �S@     �Q@     �R@     @U@     �Q@     @T@      R@     �L@     �T@      T@     @Q@      Y@      R@     �V@      V@      Z@      X@     �[@      [@      Y@     �U@     @V@     �^@     �_@     @Z@     @]@     �\@      `@      Y@     �a@     �]@     �`@     �c@      `@      a@     �d@     �a@      d@      e@     �c@     �h@      g@     �g@     @i@     �g@     �j@     0t@     �o@     `j@     �m@     �k@     �m@     �l@     pp@     �p@     0p@     �q@     `p@     q@     �t@     @t@      s@     �u@     �u@     t@     �t@      v@     �v@     �x@     px@     pz@     |@      x@      ~@     ��@     `{@     0}@     �}@     @~@     �}@     @     h�@     �@     ��@     p�@     X�@     P�@     X�@     Є@     ��@     X�@     Є@     (�@     ؇@     �@     �@     ��@     ��@     X�@     ؋@     Ќ@     ��@     `�@     ��@     ��@     ��@     ��@     H�@     �@     �@     ��@     P�@     �@     �@     �@     D�@     ��@     �@     ��@     <�@     d�@     "�@     ��@     ��@     r�@     Ʀ@     �@     ��@     ��@     ��@     ��@     ��@      �@     f�@     Ѱ@     >�@     �@     ��@     �@     ��@     ��@     D�@     �@     W�@     ��@     ��@     B�@    ��@     _�@     g�@    �y�@    �/�@     ��@    �a�@     G�@    �0�@     ��@    �K�@    �v�@     V�@    �S�@    ��@     �@      �@     ��@    ���@     ��@     ��@    ��@     �@     9�@     �@     ε@     ��@     ֱ@      �@     ��@     ��@     ڰ@     y�@     `�@     �@     d�@     ��@     A�@     �@     B�@     D�@     �@     �@     |�@     Q�@     4�@     ��@     ,�@     ��@     x�@     �x@      u@     p|@     `r@     �s@     �t@     x�@     �t@        
�
predictions*�	   �4K�   ���@     ί@!  `��E8�)d݁h�Z@2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9��f�ʜ�7
������pz�w�7��})�l a��_�T�l�>�iD*L��>f�ʜ�7
?>h�'�?�T7��?�vV�R9?�.�?ji6�9�?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@{2�.��@!��v�@�������:�               @      �?      �?      @      @      @      "@      $@      *@      2@      :@      5@      B@     �C@     �G@      M@      Q@     �Q@     �N@     �S@     @\@     �Y@      Y@      [@     �\@     �^@      W@     @]@      W@     �V@      V@     @V@     �R@     @V@     �T@      Q@     �F@      L@     �H@      C@      C@     �B@      =@      <@      4@      9@      4@      6@      .@      3@      .@      (@      4@      *@      (@      @      .@      "@      @      @      $@      "@      @      @      �?      @       @      �?      �?      @      @       @      �?      @      �?      �?       @      �?      �?      �?      �?      �?              �?      �?       @              �?              �?              �?              �?              �?               @               @              �?      �?       @      �?       @      �?      �?       @      @       @      �?      @       @       @      @              @      @      @      @      @      @      $@      *@      @      @      $@      "@      "@      *@      $@      @      0@      5@      4@      6@      @@      8@      9@      >@      ?@      3@      >@      8@     �@@      @@      6@     �@@      :@      ;@      A@     �A@      @@      C@      ;@      E@      C@     �A@      =@      9@      9@      <@      8@      =@      3@      0@      @@      5@       @      1@      (@      (@       @      1@      @      ,@      &@      .@      $@      @      @      @      @      @      @      �?      @      @       @      @      @              �?      �?      �?              �?              �?        ��}��3      �gw	�/�p���A,*�g

mean squared errorV۲<

	r-squared�D?
�L
states*�L	   �Z��    ��@    c2A!AX��p͹�)����5A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             8�@     ��@     ԓ@     x�@     ��@     0�@     0�@     ��@     h�@     l�@     l�@     ��@    ���@     4�@     ��@     ��@     e�@     E�@      �@     ��@     _�@     �@     I�@     h�@     ��@     ��@     H�@     L�@     
�@     z�@     �@     G�@     ݴ@     ۶@     ��@     ��@     ��@     ��@     ��@     `�@    ��@    ���@    �Z�@    �>�@     ��@    �W�@    �8�@    ���@    �Q�@     ��@    �#�@    �V�@     ��@     3�@    ��@     \�@    ���@    ���@     ��@    ���@    �,�@    ���@    �g�@    ���@     /�@     ׻@     ͹@     ��@     õ@     "�@     �@     8�@     T�@     �@     8�@     ,�@     �@     ��@     ̦@     z�@      �@     �@     ��@     ��@     ��@     4�@     �@     �@     ��@     �@     Ԛ@     ��@     <�@     �@     ��@     ��@     l�@     \�@     l�@     ��@     ��@     ��@     ؏@     �@     X�@     h�@     0�@     ��@     ��@     (�@     x�@     ��@     ��@     ��@     0�@     ��@     p�@      �@     �@     �@     H�@     �@     ��@     ؁@     ��@      �@      �@     h�@     Ђ@     �@     ��@     0@     �}@     @~@     �|@      |@     �y@      z@     �{@     �y@     �x@     �{@     �w@     �x@     �x@     `t@     Pw@     �w@     �u@     �u@     0u@     �s@     @t@     t@      r@     Ps@      s@     pr@     �|@     �o@     Pr@     pq@      p@      o@     �q@     �u@     @n@      n@     @n@     �k@      k@     �j@     �k@     �h@     �k@     �k@     �g@     �g@     @h@     `d@     @h@     �h@     `h@     �e@     �f@      d@      f@     �b@     `d@      e@     �d@     �b@     �c@     �c@     @a@      _@     �`@     �]@      `@      `@     �_@      b@      [@     �]@     @X@     @Z@     @\@     �Y@      Z@     @Y@     @\@     �U@     �Y@      R@      Z@     �X@     �U@      Z@     @T@     �X@     @V@     �W@     �V@     �S@     @P@     �T@     �P@      Q@      J@     @Q@      S@      Q@     �Q@     �Q@     @Q@     �N@     �N@     �Q@      P@      M@      K@      L@     �K@     �N@     �E@      N@     �H@     �F@      N@      C@     �F@     �I@     �F@     �D@     �I@     �K@     �I@      H@     �B@     �C@     �D@      ?@     �G@      ?@      5@      H@     �@@      >@      :@      A@     �E@     �@@      :@      <@      =@      4@      5@      4@      3@      2@      ;@      =@      8@      ;@      :@      (@      9@      0@      3@      6@      1@      4@      2@      2@      0@      3@      .@      3@      0@      *@      (@      *@     0�@     @�@      $@      @       @      @      $@      @      *@      $@      "@      0@       @      $@      "@      $@      "@      .@      (@      *@      $@      &@      ,@      ,@      (@      .@      5@      "@      2@      (@      (@      *@      .@      "@      .@      *@      $@      1@      2@      $@      ,@      3@      7@      8@      3@      7@      2@      $@      :@      =@      :@      >@      3@      5@      7@     �A@      <@      >@      B@      8@      ;@      :@      7@      B@      @@      :@     �@@      7@      7@     �E@     �A@      >@      E@      >@      C@      >@      ?@      M@      F@      @@     �E@      I@      H@     �G@      I@      I@     �H@     �G@      L@      N@      K@     �O@      Q@     �M@      K@      G@     �Q@      M@     @P@     �P@      Q@     @Q@     �P@     @P@     �R@     �T@     �U@     �W@     �S@      X@     @U@      W@     @S@      W@     �V@     @Y@     �[@     �W@     @Z@     �Y@     @W@     @^@     @^@     �[@     �^@      `@     �[@     @^@     �^@     �a@     �c@     �a@      a@     @c@     @e@     �c@     �e@     @e@     @b@      d@     `h@      h@     �h@     �f@     �k@     @h@     �v@     �l@     �j@     `n@     `p@     `o@     @p@     q@     �q@      t@     �o@     pq@     r@     �r@     u@      u@     �t@     Pu@     `w@     Pw@     0w@     �v@     �y@     @z@     `y@     `z@     �{@     �y@     @{@     `{@     �}@     �@     P~@     X�@     x�@     �@     x�@     �@     0�@     ��@     ��@     ��@     ��@      �@     �@     ��@     �@     8�@      �@     ��@     ��@     8�@     ؊@     �@     0�@     ��@     ��@     ,�@     ��@     ��@     ��@     �@     �@     `�@     ��@     �@     ��@     8�@     ��@     ��@     h�@     h�@     x�@     h�@     ��@     *�@     ��@     \�@     �@     �@     �@     f�@     �@     &�@     ��@     .�@     ư@     �@     N�@      �@     4�@     T�@     R�@     #�@     �@     ��@    ���@    ���@    ��@     ��@     b�@    ���@    �l�@    ���@    ���@    ��@    �;�@    ���@    ���@     ��@     ��@     ��@     �@    �r�@    ���@    �o�@    ���@    ���@    �,�@     2�@    �m�@     ��@     3�@     A�@     ��@     ��@     �@     �@     `�@     ?�@     �@     ��@     ߱@     v�@     �@     ��@     ��@     �@     ϲ@     �@     ��@     
�@     ��@     K�@     2�@     G�@     �@     P�@     ��@     ��@      y@     �s@     �~@     �t@     Pq@     ps@     Ђ@     0u@        
�
predictions*�	   @�k׿    �@     ί@!  �a{@I@)���U�W@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
���Zr[v�>O�ʗ��>6�]��?����?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�               @      @      @      @      $@      &@      .@      1@      ;@      9@      A@      D@     �H@      D@     �J@     �I@     �M@     �P@     �S@      Q@      R@     �T@     �U@      V@     �R@     �P@      T@     �Q@      R@     �P@      G@     �O@      J@      K@      H@     �A@      >@      8@      @@      :@      ;@      9@      ;@      .@      <@      8@      2@      ,@      *@      "@      @      @      "@      @      $@       @      @      @       @      @      @      @      �?      @      @      @      @       @      @       @              �?      �?      �?              �?      �?       @       @      �?               @              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?      @      �?       @              �?      @      �?      �?              @      �?       @      @      @      @       @       @      @       @      @      @       @      @      @       @      "@      @      &@      @      @       @      0@      .@      6@      1@      *@      2@      6@      .@      3@      <@      ;@     �@@      B@     �A@     �C@     �A@      H@     �H@     �A@     �O@     �G@      G@     �N@      M@     �L@      J@     �P@      I@      K@     �I@      N@      G@     �J@      F@      J@      D@      B@      B@      ?@      7@      >@      5@      8@      4@      .@      *@      4@      @      (@      "@      $@      @      2@      @      @      @      @      "@      @      �?      @       @      @      �?       @       @      �?      �?              �?              �?        Kݧ�3      Y2Y	���p���A-*�g

mean squared error ��<

	r-squared��	?
�L
states*�L	   ���   ���@    c2A!s�0!Ű�)�Q)�yA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Ј@     `�@     H�@      �@     ��@      �@     0�@     ��@     x�@     ��@     d�@     N�@     )�@     ��@     ��@     ��@     "�@     ��@     ]�@     M�@     ��@     �@     �@     J�@     {�@     ��@     ��@     ?�@     e�@     U�@     ܲ@     Z�@     ��@     �@     D�@     	�@    ���@    ���@    �P�@    ���@     g�@     L�@    �=�@    ���@     �@     b�@     ��@    ���@     ��@    �2�@     S�@     ��@     �@     ��@    ���@    �;�@    ���@     ��@    ���@     !�@     ,�@    ���@    ��@     ��@     ��@     c�@     �@     ��@     ��@     ]�@     X�@     �@     ��@     �@     ު@     �@     :�@     ��@     ,�@     n�@     ��@     ܢ@     >�@     p�@     h�@     p�@     4�@     ,�@     �@     ܚ@     �@     ��@     ��@     @�@     D�@     p�@     �@     ��@     ��@     ��@      �@     4�@     ��@     ��@     ��@     ��@     0�@     p�@     �@     ��@     x�@     ��@     ��@     H�@     ��@     ��@     ��@     Є@     ��@     ��@     �@     `�@     ��@     �@     ؁@     ��@     ��@     �@     ��@     X�@      }@     �@     �|@     P~@     �}@     �x@     �{@     `y@     �y@     �x@     �x@     �w@      v@     pv@     Pw@      x@     �t@     �u@      u@     Ps@      s@     �s@     0t@     �s@     �q@     �s@     �z@      r@     �q@      p@      o@      q@     @n@     �n@      l@     �n@     `m@     z@     `o@     �i@      j@     �l@      o@     @k@     @i@     �i@     �f@     �f@     @g@     �g@     `f@      j@     `g@     @b@     �c@     �c@     �c@     �g@     @b@     `b@     �c@      d@      b@      a@     �a@     �b@     @b@     �`@     �a@     @^@     �]@     �[@     @]@      `@      [@      U@     �Y@      W@     �\@     �W@     @P@      Z@     @X@     �U@      R@     �W@     @Y@      U@     �U@     @V@     @U@     �V@     @U@      W@      T@      Q@      Q@      W@      O@      T@      O@     �O@     �J@      J@      M@      K@      I@     �R@      R@     �N@      T@     �H@      M@     �I@      M@     �P@      M@     �L@      B@     �B@     �C@     �F@     �I@     �D@     �G@      E@      E@     �G@      ?@      I@      ?@     �E@     �@@     �@@      B@      F@      ?@      8@      >@     �D@      @@      =@      <@     �D@      5@      <@      7@      A@      A@      3@      4@      4@      8@      4@      2@      7@      3@      1@      3@      2@      0@      .@      9@      0@      &@      9@      *@      ,@      6@      *@      .@      :@     \�@     ��@      @      (@      (@      &@      @      &@      *@       @      $@      @      2@      @      @      3@      @      .@      "@      1@      *@      *@      ,@      ,@      @      2@      *@      0@      *@      *@      $@      1@      5@      2@      3@      ,@      0@      6@      5@      $@      3@      5@      0@      8@      1@      =@      7@      ,@      2@      ?@      ;@      3@     �@@      ?@      ?@      4@      A@      7@      9@      @@      1@      7@      8@      ;@      3@      8@      ?@     �A@      C@      B@     �H@      E@     �A@     �A@      >@     �C@      D@      B@      I@      G@      E@     �J@     �G@     �H@      H@      I@     �F@      G@      G@      E@     �J@      Q@      H@     @P@     @P@     @S@     �P@     �Q@     �O@     �R@     �P@     �S@     �S@      R@     �P@     �T@     �P@     �P@     @T@     �Q@     �R@      W@     @U@     �W@      W@     �U@     �X@     �W@     @\@     @Z@     �[@     @X@     �\@     @Z@     @Z@     �^@      c@     @_@      `@     �b@     �_@      a@     �c@      `@     @d@      c@     @c@     `e@     �e@     �d@     `f@     �g@     �h@     �h@      i@     @m@     `k@     �m@     Pv@     @p@     �n@     @n@      p@     @n@     @p@     �r@     t@     �p@     `s@     �s@     u@     @t@     Ps@     �s@     @u@     u@     @v@     �w@      w@     �w@     �v@     �x@     �{@     p{@     �{@     �z@     �|@     �~@     �}@     �@     �@     h�@     ��@     ��@     p�@     ��@     h�@     �@     ��@     ��@     ��@      �@     �@     ��@     P�@     ȇ@     x�@     �@     ��@      �@     ��@     ��@     (�@     ��@     ��@     Ȑ@     H�@     �@     ��@     ��@      �@     ��@     0�@     ܗ@     (�@     ��@     �@     ,�@     ��@     \�@     |�@     ��@     Ģ@     h�@     f�@     >�@     ��@     ��@     �@     ��@     ܫ@     �@     ��@     \�@     :�@     �@     :�@     ��@     �@     ]�@     X�@     �@     ��@     ��@    ���@     :�@     �@     ��@     ��@     ��@     ��@     ��@     ��@    �O�@     �@    ���@    �]�@    ���@     @�@     G�@    ���@    ��@     ��@    �K�@     
�@    ���@     ��@     ��@     ��@     ��@     ��@     Q�@     ڳ@     ز@     Ա@     =�@     ��@     A�@     �@     ѱ@     ��@     a�@     ��@     ��@     7�@     R�@     ѵ@     ��@     l�@     <�@     k�@     U�@     ��@     �@      �@     ��@      x@     �u@     `|@      u@     ps@     �t@     H�@     �s@        
�
predictions*�	   �(�   ���@     ί@!  �a+!�)<�9�C�Y@2�������2g�G�A�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"��vV�R9��T7����5�i}1���d�r�x?�x��f�ʜ�7
������1��a˲���[���uE����>�f����>8K�ߝ�>�h���`�>�T7��?�vV�R9?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@�6v��@�DK��@{2�.��@�������:�              �?              @      �?      �?      �?      @      @      �?       @       @      &@      &@      C@      9@      D@      E@      I@     �I@     @Q@     �I@     �Q@     �O@     @V@      T@     �U@     �S@     @V@     �Y@     �R@     @W@     �R@     �U@     �X@     �S@     �T@     @R@      J@      L@      N@      M@      N@     �F@      J@      ?@     �G@      6@      =@      5@      3@      0@      8@      0@      3@      *@      $@      ,@      *@      "@      @      "@      @      @       @      @      $@      @      "@       @      �?      @      �?       @      @      @      �?      @      @       @      �?      @      �?       @      �?              @              �?      �?              �?              �?              �?              �?              �?              �?               @       @               @      �?              �?       @              �?      @              �?      @      @       @      �?      @       @      @      @      @      @      &@       @      @      .@      ,@      $@      "@      &@       @      *@      .@      0@      ;@      7@      4@      8@      ,@      6@     �@@      =@      7@     �B@     �E@     �A@      A@      D@      E@      B@      D@      K@     �H@      H@     �F@     �I@      E@      ;@      A@      5@      ?@      B@     �B@      <@      <@      >@      A@      7@      0@      1@      3@      0@      &@      .@      "@      &@      0@       @      &@       @      &@      @      @       @      @      @      @      @       @      @      @      @       @              �?              �?      �?              �?        ��3      ��<	"�p���A.*�f

mean squared error�J�<

	r-squared�	?
�L
states*�L	    ;��    �@    c2A!B�$�u��):���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ؎@     ��@     4�@     ؇@     p�@     ��@     ��@     <�@     ��@     `�@     J�@     ��@     ��@     x�@     ��@     ��@     m�@     W�@     3�@     Z�@     n�@     &�@     ¯@     �@     �@     .�@     r�@     �@     z�@     �@     !�@     �@     }�@     ��@     �@     ��@    ���@    �X�@     ��@     &�@    �'�@     2�@     7�@    ���@    ���@    ��@    ���@     ��@     6�@    ���@     2�@     7�@     e�@    �n�@    ���@     [�@     ��@     g�@     �@     q�@     ��@    ���@     ��@     <�@     �@     y�@     7�@     �@     ��@     ��@     ��@     ��@     �@     ��@     6�@     ��@     B�@     ��@     �@     �@     R�@     ؠ@     4�@     L�@     \�@     ��@      �@     �@     �@     x�@     h�@     |�@     ��@     ��@     Ԕ@     �@     �@     ��@     ؒ@     T�@      �@     Ў@     ��@     x�@     h�@     ��@     P�@     Ȋ@     ��@     �@     �@     ��@     H�@     ��@     p�@     H�@     @�@     ��@      �@     ��@     ��@     ��@     �@     X�@     P�@     �~@     ��@     �@     �~@     0@     �|@     �{@     0}@     �z@      {@     �|@     0z@     {@     �y@     0y@     �w@      x@     �v@     v@     �v@     �v@     Pu@     `v@     `v@     Pt@      r@     0t@     �y@      s@      s@      p@      r@     �o@     Pq@     �o@     �n@     �n@     `n@     �o@      i@     @l@     �m@     �l@     `o@     Pw@      o@     `k@     �j@      l@     �f@     �k@      i@     �h@     �f@     @d@     �e@     @b@     �e@      h@     �d@     @f@      `@     `c@     @a@      b@      b@      ]@     @a@     �^@     @d@     �`@     �`@     �a@     �\@     @Z@     �[@     �X@     ``@     �Y@     �V@     �Z@     �W@     �[@     �[@     @W@      Z@     �T@      S@     �U@     �Y@     �R@     �U@      Y@     �U@     �R@      R@     �T@     @S@     �Q@     @U@     �P@      V@     �Q@      U@     �M@     �N@     �P@      Q@      Q@     @P@      O@      K@     �P@     �J@     @P@     @S@      N@      J@     @P@      L@     �L@     �I@     �F@      G@     �D@      H@      H@     �H@      B@     �F@     �B@      @@      A@     �@@      B@      A@      9@      C@      >@     �B@     �A@      @@      >@      =@      :@      B@      A@      =@      :@      9@      7@      1@      C@      8@      5@      6@      9@      9@      :@      3@      6@      0@      7@      9@      (@      .@      1@      1@      2@      1@      .@      3@      :@      .@      4@      @     h�@     H�@      @      &@       @       @       @       @      $@       @      @       @      "@      ,@      &@      "@      *@      &@      "@      &@      ,@      (@      (@      $@      $@      4@      .@      *@      *@      .@      .@      &@      *@      5@      0@      .@      1@      1@      3@      4@      9@      5@      5@      :@      ,@      7@      ;@      3@      8@      9@      2@      4@      *@      6@     �@@      <@      4@      5@      9@     �A@      <@     �@@      7@     �@@      A@     �B@      @@     �@@      <@      9@      B@      B@     �@@     �G@     �C@     �H@      A@      L@      K@     �E@      D@      D@     �D@      @@     �F@     �D@      I@     �L@     �E@      C@     �J@     �I@      Q@      P@      N@     �P@     �S@     �J@      Q@     @T@     �R@      K@      R@      S@     �U@      Q@      V@     �R@     �S@     �T@     @V@     �R@     �T@     �R@      T@     �T@      Y@     @Z@     �Z@     �U@     �Z@     �Y@     @[@     �X@     �Z@      ]@     @a@     @`@     �a@      c@      b@      c@      `@      a@      d@     �c@     @h@     �d@     `c@     �d@     �k@      i@     �g@     �g@     �j@      k@     �l@     �l@     �m@     �u@     @n@     �m@     �p@      n@     �p@     �q@     �p@     p@     �q@     �r@     �q@     ps@     �r@     �q@     0t@     �s@      w@     `v@     Px@     `w@     Py@     Py@     `z@     �w@     y@      {@     �z@     �}@      @      @     x�@     ��@     @@     (�@     x�@     �@     �@     @�@     ؂@     ��@     ��@     Ȅ@     ��@     �@     ��@     ��@     �@     �@     8�@     h�@     �@     �@     ��@     `�@     ��@     ��@      �@     $�@     ��@     ��@     ��@     D�@     �@     ��@     ��@     `�@     ��@     d�@     ��@     ��@     T�@     ��@     ��@     
�@     أ@     .�@     �@     X�@     �@     0�@     8�@     �@     Z�@     �@     �@     �@     ޵@     k�@     �@     ɺ@     ��@     ʾ@    ���@    �%�@     ��@    �*�@     2�@    ��@     ��@     �@     +�@    �4�@     ��@     ��@     M�@    ���@    ���@    �t�@    �C�@    �L�@    �m�@     ��@     ��@     ��@     �@     ��@     �@    �4�@     +�@     ��@     2�@     �@     �@     ��@     ű@     ��@     �@     ��@     �@     ��@     ��@     �@     F�@     =�@     ��@     �@     ��@     Ǹ@     ��@     ��@     Y�@     y�@     v�@     D�@     ؎@     �@     ��@     �{@     ~@     @u@     �p@     px@     ��@     @o@        
�
predictions*�	    ���   `bk@     ί@!  p�R�/�)��Z�V@2�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6������6�]����FF�G �>�?�s���O�ʗ���})�l a�>pz�w�7�>��Zr[v�>O�ʗ��>6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?�.�?ji6�9�?�S�F !?�[^:��"?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�Š)U	@u�rʭ�@�������:�              �?       @      �?      �?               @      @      .@      &@      "@      2@      @@     �G@     �G@      G@      L@      O@     �Q@      P@     @X@     @V@     @V@     �S@     @Z@     �Y@     �Y@     �Z@     @V@      U@     @V@     �S@     �V@      T@      N@      P@     �O@      G@      K@     �H@     �I@      F@     �D@      :@      9@      =@      8@      8@      2@      :@      6@      0@      &@      $@      1@       @       @      @      @      @      @      @       @      @      @      �?      @      @      @      @      @      @      �?      @       @               @       @       @              �?              �?      �?              �?              �?              �?              �?      �?       @              �?      �?      �?              �?      �?       @       @              �?       @              @      �?               @      @      �?      @      @      @      @      @      @      @      @      @      "@       @       @      $@      &@      0@      $@      6@      5@      0@      @@      3@      9@      4@      3@      <@      =@      C@     �A@      @@     �C@      ?@     �C@     �B@     �@@      :@      G@     �H@     �F@      9@     �@@     �C@     �G@      E@     �A@      @@      E@      A@      @@      9@      :@      :@      ,@      8@      4@      7@      ,@      &@      *@      .@      @      @      @      $@      @       @      @      @      @       @       @       @       @      @       @       @       @              �?              �?      �?              �?        ef�x�2      {�T	��q���A/*�e

mean squared errorC�<

	r-squaredԸ?
�L
states*�L	   �'��    ��@    c2A!��4����)^i�qA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             P�@     ��@     ,�@     4�@     ��@     `�@     ؆@     H�@     Џ@     @�@     x�@     Φ@    ���@     ��@     �@     ��@     �@     ��@     +�@     ʱ@     ذ@     5�@     �@     ��@     ��@     �@     ��@     p�@     ��@     ��@     �@     }�@     0�@     5�@     (�@     j�@     ܿ@    �M�@    ���@     �@    ���@     ��@    ���@     (�@     ��@     ��@     "�@     R�@    ���@     �@    �l�@    ��@    �H�@     �@    ��@    ���@    ��@    �j�@    ���@    ��@     ��@    ���@    ���@    �G�@     ��@     ��@     ¸@     Ͷ@     ��@     t�@     b�@     O�@     ��@     Z�@     ��@     j�@     ��@     ئ@     �@     ��@     D�@     ��@     ��@     p�@     ��@     �@     Ԝ@     Ț@     ��@     ��@     l�@     ș@     �@     �@     ��@     ��@     ��@     ��@     ؒ@     ��@     0�@     �@     P�@     ��@     `�@     ��@     �@     ��@     ��@     (�@     ��@     `�@     ��@     0�@     ��@     ؆@     ��@     ��@     P�@     H�@     H�@      �@     Ȃ@     (�@     ��@     p�@     ��@     0�@     (�@     P~@     ��@     �|@     0}@     0~@     �z@     �z@     �x@     P|@     �z@     �x@     �w@     pv@     �u@     Px@     px@      v@     �w@     Pw@     �u@     �t@     �s@     �r@     @~@     0u@     r@     ps@      t@     0p@     �r@     @r@      q@      q@     �k@     0p@     pp@     �j@     �o@     @k@      l@     `k@     �m@     �j@     pr@     `r@     �i@      i@      h@     `g@     �f@     �f@      f@     @e@     @e@     @f@     �d@     `e@     `c@     �c@     `b@     @b@      b@     �d@     �a@      _@      c@     �^@     �`@      `@     �`@     �^@     ``@     @^@      ^@     �^@     @]@     �]@      \@      ]@     �Z@      V@     �U@     @Z@     �U@     �[@     �\@     @Z@     @Y@     @\@     �V@     �V@     @Q@     @U@      U@     @R@     �S@      R@      S@      F@      S@      R@      U@     �R@     �O@     �L@     �Q@     �K@     @S@     �I@      Q@     �O@     �J@     �J@     �H@     �M@      J@     �I@      L@     �H@      G@      K@      L@      C@      F@     �C@      H@      H@      <@     �H@     �A@      A@      >@      K@      H@     �A@     �E@      2@     �A@      ?@      4@      ;@      7@      ?@      B@      4@      7@     �A@      <@      =@      >@      3@      6@      3@      3@      2@      8@      9@      5@      5@      9@      6@      6@      3@      1@      9@      5@      6@      3@      3@      7@      @      ,@      8@     ��@     8�@      (@      �?      @      (@      ,@       @      $@      $@      @      @      (@      @      .@      ,@      "@      *@      *@      *@      &@      (@      &@      8@      *@      *@      (@      ,@      *@      ,@      4@      &@      *@      :@      0@      4@      6@      @      .@      4@      8@      2@      1@      4@      4@      1@      3@      4@      ?@      :@      ;@      5@      9@      4@     �@@      <@      9@      <@     �@@      3@      =@      F@      7@     �D@      ;@      .@      :@     �E@      <@      B@      C@     �D@     �D@      E@      B@     �B@     �A@      B@     �C@     �F@     �E@      M@      G@      K@     �A@     �B@     �I@     �G@     �N@     �K@     �K@      L@      N@     �H@     @P@      S@     �O@     �N@     �K@     �P@     �M@     �M@      R@      O@      O@     �T@     @T@     @Q@     @T@     @U@     �T@     �X@     @T@     @V@     @W@     �V@     @W@      W@     �X@      V@      Y@     �Z@     �Z@     �Z@     �Z@     �a@     �\@      ]@     �\@      [@     �a@      ^@      `@     �b@     �`@     �b@     �a@      c@     @e@     �b@     �d@     `g@      i@      i@     �i@     �k@     �j@     �g@     `i@     �m@      r@     �q@     �o@     �p@     �p@     �p@     `n@     �p@     r@     �q@     pr@     Pp@     pu@     �s@     @r@     0u@     ps@     v@     �t@      v@      x@      x@     pz@      {@     P{@     z@     �{@     `}@      }@     �~@     �~@     ~@     ��@     ��@     �@     ��@     8�@     P�@     @�@     0�@      �@     �@     8�@     `�@     @�@     І@     ��@     `�@     ��@     ��@     ��@     `�@     p�@     P�@     X�@     4�@     ��@     d�@     ��@     ē@     ��@     ��@     8�@     ��@     P�@     �@     Ț@     �@     ܜ@     �@     H�@     ��@     Ρ@     �@     �@     �@     �@      �@     ̧@     �@     ��@     F�@     ��@     �@     �@     V�@     �@     ^�@     չ@     Y�@     "�@     o�@     ��@    ���@     l�@     W�@    �Q�@    ���@     ��@     !�@    ��@    ���@     w�@     �@     ��@    ��@    ��@    ���@     ��@    ���@    ���@    �K�@    ���@     ��@     ��@     1�@     .�@     �@     ��@     ��@     �@     ߴ@     Y�@     [�@     ��@     ��@     ��@     ɱ@     б@     β@     *�@     ձ@     ��@     `�@     l�@     +�@     Ե@     �@     ��@     �@     _�@    ���@     P�@     4�@     ��@     �@     ��@     �@     �@     �v@     Pr@     �|@     h�@     @h@        
�
predictions*�	   @�p�    �V@     ί@!  @�::9@)�^�]Z@2�2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����d�r�x?�x���S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@�DK��@{2�.��@�������:�              �?              �?      �?      @       @       @      @      @      @      @      2@      0@      ;@      4@      @@     �E@      E@      I@     �H@     �I@     �Q@      R@     �R@     �R@      S@     �S@      T@     �W@      [@     �Y@     �R@     @T@     �S@      P@     @Q@     �R@     �K@      F@      F@      C@      D@      C@     �C@      >@      D@      @@      9@      5@      2@      7@      1@      3@      &@      0@      ,@      $@      $@      &@      &@      @      $@       @       @      @       @      @      @      @      @      @               @              @      �?              @              �?       @       @      �?              �?       @               @              �?               @              �?      �?              �?      �?              �?      �?      �?              @      �?       @      @       @      @      @      @       @      @      @       @      @      @       @      @      (@      (@      "@      .@      0@       @      1@      &@      *@      .@      8@      6@      2@      2@     �A@      2@      ?@      ;@      C@      E@      >@     �@@      =@     �I@      E@      K@     �B@     �F@     �M@     �K@      K@     �C@     �@@      E@     �A@      H@      J@      <@      D@     �J@      <@     �G@      A@      8@      7@      C@      ?@      5@      0@      "@      .@      @      *@      .@      @      @      (@      @      @      @      @      @              @      @      @      @      �?      �?      @              �?              �?        }w��3      �gw	�C.q���A0*�g

mean squared errorem�<

	r-squared�>
?
�L
states*�L	   ����   ��@    c2A!e�jLћ��)K"����A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     �@     Џ@     ��@     p�@     ��@      �@     ��@     ؔ@     Ĝ@     �@     +�@     u�@     ��@     ��@     P�@     >�@     O�@     ��@     ��@     ��@     ү@     b�@     ��@     Ю@     �@     ��@     C�@     ��@     �@     ݳ@     ��@     �@     Y�@     ļ@     ��@     ��@    �j�@    �O�@    ��@     ��@    ���@    ���@    ���@    ��@    �y�@    �2�@    ���@     ��@     ��@    ���@    �Q�@    ���@    ���@     l�@     ��@    ��@    ���@     ��@     ��@    ���@     _�@     ��@     ��@     >�@     d�@     ۶@     ̴@     �@     
�@     �@     J�@     �@     ��@     ~�@     *�@     ަ@     J�@     ��@     ��@     �@     |�@     6�@     X�@     (�@     �@     �@     (�@     �@     |�@     ��@     ��@     ��@     ��@     4�@     \�@     �@     �@     �@     ��@     А@     �@     x�@     0�@     ��@     0�@     �@     ��@     ��@     ��@     \�@     (�@     ��@     ��@     ��@     ��@     ��@     ��@     H�@     ��@     ��@     ؂@     ��@     ��@     ��@     ��@     X�@     P~@     0�@     �~@     �{@     �~@     �z@      }@     @{@     �x@     �|@     �|@     0}@     `{@      z@     `x@     �x@     �t@     �v@     `y@     `x@     �t@     �t@     Pv@     �u@     0s@     �q@     0t@     �|@     0s@     �r@      q@     �s@     �n@      q@     @n@     �o@     �k@     �i@     �n@     �i@     �l@     @m@      o@     �n@     �j@     �f@     w@     �f@     `g@      l@     �i@     �g@     �g@     �h@     `f@      g@     �c@     `e@     `c@     �b@      d@     �b@     �`@      b@      b@     `a@     `a@     �a@     `a@     �[@     @b@      _@      a@      [@     ``@     �Y@     @\@     �X@     �Y@     �]@     �X@     @^@     �X@     �Z@     �T@      X@     @Y@      [@     @V@     �W@      ]@     @T@     �V@     @U@      Z@     �T@      O@     �T@     �R@      S@     @Q@     �R@      S@      O@     �S@     �O@     �N@      O@      O@     �Q@      I@      N@      N@      H@     �R@     �H@      H@      I@      K@     @P@      I@     �F@     �D@      H@     �N@     �E@     �G@     �C@      H@     �F@     �I@      :@      F@     �A@      @@      G@      C@      <@      D@      @@      ?@      @@      7@      ?@      7@      <@      <@      ;@      9@      2@      9@      >@      9@      3@      8@      1@      2@      =@      =@      6@      :@      ,@      1@      7@      ;@      ;@      4@      7@      6@      4@      2@      0@      1@      *@     �@      �@       @      @      ,@      &@      @      &@       @      &@      @      @      (@      ,@      &@      ,@      ,@      &@      $@      1@      &@      "@      "@      &@      $@       @      $@      ,@      1@      3@      &@      7@      *@      4@      0@      "@      4@      3@      0@      "@      .@      2@      (@      2@      6@      3@      7@      6@      C@      9@      1@      5@      9@      9@      :@      6@      7@      6@     �A@      6@      A@     �A@      9@      7@      7@      <@      D@      @@      =@      <@      =@      :@     �B@      >@      D@     �D@     �E@      <@      A@     �A@     �H@     �D@      C@     �H@      C@      H@      H@     �J@     �J@      C@      J@      K@      N@      P@      J@     �O@      J@     �N@     @P@      S@      O@      O@      P@      T@     @P@     @P@     �S@     �Q@     �Q@     @X@     �U@     @V@      Y@     �S@     �V@     �W@     �T@     �W@     �W@     @Z@     �W@      X@      X@      _@     �_@     �_@     �^@     �_@     @_@     �[@     �`@     �_@     �a@     �a@      d@      c@     @g@      a@     �d@     �e@     `d@     �f@     �c@     �e@     �g@     �i@     @l@     �k@     �i@     �k@     @k@     �g@     �k@     �n@     pq@     �t@     �l@     0p@      q@     Pr@     Ps@     Pp@     �q@     pr@     �r@     0s@     �u@     pv@     �v@     Pu@     �t@     �w@     @w@     �w@     @w@     �{@     �{@     `{@     @|@     �|@     �~@     0@      �@     ��@     ��@      �@     �@     `�@     ��@     ��@     ��@     ��@     0�@     Ȉ@     `�@     x�@     P�@     ��@     8�@     �@     @�@     ��@      �@     P�@     8�@     @�@     h�@     (�@     ��@     ��@     �@     l�@     ��@     ��@     ̗@     �@     ̙@     H�@     ܛ@     d�@     4�@     Ơ@     Π@     F�@     :�@     $�@     ��@     ��@     ި@     ��@     �@     ^�@     �@     ��@     ��@     A�@     ��@     #�@     *�@     !�@     ��@     �@    �~�@    ���@    ���@     A�@    �h�@     �@    �"�@    ���@     ��@     ��@     ��@    ���@     ��@     ��@    ���@     ��@     ��@    ���@    ���@     �@     +�@    ���@    �)�@    �0�@    ���@     ��@     ]�@     t�@     ��@     µ@     ~�@     �@     �@     �@     ��@     հ@     W�@     �@     ��@     V�@     ɲ@     `�@     ��@     �@     D�@     ��@     �@     ��@     �@     A�@     >�@     ,�@     p�@     �@     ��@     0�@     (�@     P�@     (�@     X�@     0�@     �j@        
�
predictions*�	   `��    <v@     ί@!  x���F�)`xF�Y@2�������2g�G�A�uo�p�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'������?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�.�?ji6�9�?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�              �?      �?              �?      @       @      @      @      @      &@      .@      6@      ;@      =@      D@     �C@      M@     �K@     �H@     �P@      P@     �R@     �S@     �S@     @V@      U@     �V@      Z@     @X@     @V@     �S@      U@     �X@     �W@     �T@     @R@     @P@      N@     �N@     �I@     �E@     �J@      E@     �E@     �B@     �E@      <@     �B@      =@      5@      5@      3@      1@      &@      2@      1@       @      .@      ,@      "@      @      "@      @       @      @      �?       @       @      @       @      @       @      @      �?              �?      �?       @      �?      �?      �?      �?              �?              �?       @              �?              �?              �?              �?      �?              �?              �?              �?              �?               @      �?              �?      �?      �?       @      �?      �?      @       @      �?      @      @              �?      @       @      @      @      @      @      @      @      &@      @      ,@      ,@      ,@      *@      @      @      9@      .@      3@      7@      0@     �@@      @@      @@      B@     �A@     �B@      A@      ?@     �E@      C@      @@     �D@      D@      A@      C@      H@     �F@      ?@     �C@     �C@     �A@      4@     �@@      6@      @@      =@      2@      4@      9@      3@      6@      ;@      4@      3@      1@      *@      @      @      (@       @      $@      "@      @      @       @       @      �?       @      @      �?       @      �?       @       @       @       @              �?              @              �?        �^|�B3      �%��	�Iq���A1*�f

mean squared error��<

	r-squared8?
�L
states*�L	    ���   ���@    c2A!{es��)N��2��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     �@     ��@     ��@     ��@     P�@     x�@     P�@     |�@     <�@     �@    ��@     �@     �@     )�@     ��@     y�@     1�@     ��@     ܰ@     ��@     ��@     �@     Я@     ��@     &�@     ��@     �@     K�@     �@     ��@     ��@     R�@     t�@     ��@     ѿ@    ���@     ��@    ���@    �m�@     ��@    �N�@     �@    ���@     h�@     ��@     j�@    �c�@     ��@    ���@     ��@    ���@    �}�@     L�@     ��@    �q�@    �6�@     V�@    �i�@     ��@     �@    ���@     Z�@     ��@     H�@     9�@     ڷ@     Y�@     x�@     �@     <�@     c�@     �@     t�@     ��@     �@     ̧@     F�@     |�@     �@     ��@     N�@     6�@     ޠ@     ܞ@     ��@     �@     `�@     ��@     �@     ,�@     ��@     h�@     4�@     H�@     |�@     В@     t�@     ��@     p�@     p�@     |�@     (�@     ��@     Ў@     ��@     ��@     ��@     ��@     ��@     ��@      �@     X�@     ��@     ��@     ��@     ��@     �@     Ȇ@     p�@     0�@     Ѓ@     X�@     (�@     ��@     H�@     (�@     �}@     �@     p}@     8�@     ~@     P|@     0~@      |@      {@     `{@     �|@     @}@     x@     �x@     �z@     �x@     `v@     px@     �{@     px@     �u@     pv@     `u@     �t@      u@     �r@     ps@     Pr@     �q@     �s@     �q@     `o@     u@      y@     �q@     Pq@     �n@     `o@     0p@     �l@     �j@      k@     @l@     �i@     @l@      k@     `i@     �s@      o@     `i@      e@     @i@     �d@     @g@     �e@     �c@     �e@     �c@     �d@      b@     @f@      b@     `d@      b@      c@     `c@     �_@     �`@     @a@      a@     �^@     �_@     �`@     �`@      `@     �\@     @_@      ^@     @]@     @Y@     �V@     @Z@      U@      ]@     �V@      ]@      R@      Y@     �W@     �W@     �U@     �Q@     @W@     �T@     �V@     �U@     �S@     �T@     �S@     @V@     �T@     �V@     �Q@     �R@     �P@     �O@     @P@     �Q@     �P@     @S@      N@     �P@      Q@     �J@      Q@      Q@      J@      Q@     �M@     @P@     �N@     �H@      M@     �C@      J@      D@      E@      M@      D@      C@      C@      J@     �C@      B@     �G@      >@     �F@      7@      C@     �C@      ;@      D@      C@      @@      @@      @@     �B@      8@      3@      :@      :@     �C@      :@      5@      @@      8@      4@      3@      @@      6@      ;@      7@      <@      2@      2@      9@      @@      0@      4@      4@      8@      8@      1@      5@     ��@      �@      (@      @      ,@      &@       @      @      .@       @      "@       @      @      &@      &@      (@      "@      "@      $@      &@      &@      ,@      .@      &@      3@      3@      2@      ,@      @      "@      .@      3@      ,@      &@      .@      $@      1@      4@      &@      4@      2@      1@      .@      6@      7@      7@      8@      2@      5@     �@@      ?@      =@     �@@      6@      6@      7@      5@      7@      4@      1@      <@      ;@      7@      B@      @@     �A@     �A@      6@      A@      @@      ;@     �A@      E@      >@      ;@     �E@      E@      A@     �D@     �G@     �M@      B@      H@      @@      E@     �B@     �H@     �C@      K@      K@      N@      L@     �K@     �M@      M@     �K@     @Q@     �P@     �Q@      H@      R@     �R@     �R@      O@      O@      T@     �V@     �S@     �Q@     �R@     �R@     �R@     @S@     �X@     �T@     �V@      V@     �W@     @\@      Z@     @Y@     �Y@     @Z@     �Y@      [@      a@     �_@     �`@     �`@      ]@     @b@      `@     �c@     `c@     @b@     �a@     �a@      f@      e@     �e@     `d@     �f@      e@     @f@     �j@     �f@     �f@     �i@      i@     �j@      n@     �i@     �n@     pp@      o@     `o@      q@     `r@     `v@     Pu@     pp@      r@     �s@      t@     @s@     �t@     0u@     `r@     �v@      w@     �t@     �u@     Px@     �x@     �z@     y@     �{@     �y@     �y@     �{@     �@     P@      �@     0�@     `�@     ��@     ��@     ��@     �@     ��@     ��@     �@     8�@     `�@     �@     ��@     ��@     ��@     (�@     ��@     X�@     ��@     �@     Ў@     �@     t�@     |�@     h�@      �@     0�@     ��@     ĕ@     @�@     ̗@     ��@      �@     ��@     �@     ��@     �@     $�@     &�@     J�@     L�@     �@     ��@     L�@     n�@     `�@     �@     l�@     (�@     N�@     ѱ@     %�@     ��@     �@     ,�@     ��@     ��@     ��@     Y�@    ���@    �c�@    ���@    ���@    �Y�@    �g�@     ��@     S�@    �F�@     �@     1�@    �L�@    ���@    ���@     y�@    �g�@    ��@     r�@     ��@     �@    ���@     ^�@     ��@    ���@     ��@     ھ@     ��@     7�@     �@     ~�@     r�@     ϲ@     z�@     հ@     ��@     O�@     �@     r�@     ٲ@     ��@     ��@     �@     $�@     �@     ��@     #�@     O�@     �@     &�@    �m�@     ��@     \�@     ��@     p�@     P�@     ��@     Љ@     Ȃ@     ��@     ��@     ��@     �s@        
�
predictions*�	   ��o�   @�@     ί@!  ��Y@)w:�A0`@2�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8�+A�F�&�U�4@@�$��[^:��"�ji6�9���.��豪}0ڰ�������iD*L��>E��a�W�>O�ʗ��>>�?�s��>f�ʜ�7
?>h�'�?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?w`<f@�6v��@!��v�@زv�5f@�������:�              �?              �?       @              @      @              "@      @      "@      &@      5@      3@      ;@      C@      A@     �B@     �I@     �E@     �I@      L@     �Q@     �Q@     �M@      N@     �Q@     �R@      R@     @P@      P@     �P@      O@     �N@     @P@      H@     �J@      F@     �L@      G@     �C@      G@      C@     �@@      :@      B@      3@      5@      1@      .@      3@      .@      1@      (@      $@      @      "@      @       @      @      &@      @       @      @      @      @      @      @      @      @       @      �?      �?      @       @              �?       @      @              �?      �?              �?              �?              �?              �?              �?               @              @              �?      �?       @              �?               @      @              �?       @       @              @       @      �?      @      �?      �?      @               @      @      @      (@      @       @       @      $@      "@       @      0@      *@      ,@      $@      4@      5@      5@      4@      6@      7@      @@      6@     �A@      @@     �A@     �B@     �G@     �I@     �C@      O@      O@     �H@     �K@     �H@     �J@     �P@     �N@     �J@      N@     @Q@     �E@      K@     �M@      H@     �F@      G@      F@      D@      <@      F@      A@      >@     �A@      @@      5@      3@      7@      2@      0@      1@      *@      0@      .@      @      "@      &@      @      @       @      �?      @      @      @      @               @              �?              �?        4K: