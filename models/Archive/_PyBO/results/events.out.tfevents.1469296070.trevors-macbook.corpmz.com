       �K"	  �q���Abrain.Event:21	k��     D�0	g��q���A"��
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
value	B :w*
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
:���������w
P
model/pack_1/1Const*
dtype0*
value	B :w*
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
:���������w
P
model/pack_2/1Const*
dtype0*
value	B :w*
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
:���������w
P
model/pack_3/1Const*
dtype0*
value	B :w*
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
:���������w
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
 *��"?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"H  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *W�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *W�;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��
�
6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�w:	�w:	�w:	�w
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
:	�w
�
2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�w
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel/zeros2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�w
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�w
�
/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�w
�
1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�w
�
4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�w
s
.model/RNN/MultiRNNCell/Cell0/dropout/keep_probConst*
dtype0*
valueB
 *��"?*
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
:	�w
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell0/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/sub*
T0*
_output_shapes
:	�w
�
3model/RNN/MultiRNNCell/Cell0/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell0/dropout/random_uniform/min*
T0*
_output_shapes
:	�w
�
(model/RNN/MultiRNNCell/Cell0/dropout/addAdd.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob3model/RNN/MultiRNNCell/Cell0/dropout/random_uniform*
T0*
_output_shapes
:	�w
�
*model/RNN/MultiRNNCell/Cell0/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell0/dropout/add*
T0*
_output_shapes
:	�w
�
(model/RNN/MultiRNNCell/Cell0/dropout/InvInv.model/RNN/MultiRNNCell/Cell0/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell0/dropout/mulMul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell0/dropout/Inv*
T0*
_output_shapes
:	�w
�
*model/RNN/MultiRNNCell/Cell0/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mul*model/RNN/MultiRNNCell/Cell0/dropout/Floor*
T0*
_output_shapes
:	�w
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixVariable*
dtype0*
shape:
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB"�   �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *W�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *W�;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��
�
6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�w:	�w:	�w:	�w
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
:	�w
�
2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�w
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel/zeros_22model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�w
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�w
�
/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�w
�
1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�w
�
4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�w
�
0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_14model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�w
s
.model/RNN/MultiRNNCell/Cell1/dropout/keep_probConst*
dtype0*
valueB
 *��"?*
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
:	�w
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/subSub7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/max7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mulMulAmodel/RNN/MultiRNNCell/Cell1/dropout/random_uniform/RandomUniform7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/sub*
T0*
_output_shapes
:	�w
�
3model/RNN/MultiRNNCell/Cell1/dropout/random_uniformAdd7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/mul7model/RNN/MultiRNNCell/Cell1/dropout/random_uniform/min*
T0*
_output_shapes
:	�w
�
(model/RNN/MultiRNNCell/Cell1/dropout/addAdd.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob3model/RNN/MultiRNNCell/Cell1/dropout/random_uniform*
T0*
_output_shapes
:	�w
�
*model/RNN/MultiRNNCell/Cell1/dropout/FloorFloor(model/RNN/MultiRNNCell/Cell1/dropout/add*
T0*
_output_shapes
:	�w
�
(model/RNN/MultiRNNCell/Cell1/dropout/InvInv.model/RNN/MultiRNNCell/Cell1/dropout/keep_prob*
T0*
_output_shapes
: 
�
(model/RNN/MultiRNNCell/Cell1/dropout/mulMul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2(model/RNN/MultiRNNCell/Cell1/dropout/Inv*
T0*
_output_shapes
:	�w
�
*model/RNN/MultiRNNCell/Cell1/dropout/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/RNN/MultiRNNCell/Cell1/dropout/Floor*
T0*
_output_shapes
:	�w
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
:	�w
d
model/Reshape/shapeConst*
dtype0*
valueB"����w   *
_output_shapes
:
e
model/ReshapeReshapemodel/concatmodel/Reshape/shape*
T0*
_output_shapes
:	�w

model/dense_wVariable*
dtype0*
shape
:w*
	container *
shared_name *
_output_shapes

:w
�
.model/dense_w/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@model/dense_w*
valueB"w      *
_output_shapes
:
�
,model/dense_w/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *W�*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *W�;*
_output_shapes
: 
�
6model/dense_w/Initializer/random_uniform/RandomUniformRandomUniform.model/dense_w/Initializer/random_uniform/shape*
_output_shapes

:w*
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

:w
�
(model/dense_w/Initializer/random_uniformAdd,model/dense_w/Initializer/random_uniform/mul,model/dense_w/Initializer/random_uniform/min* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:w
�
model/dense_w/AssignAssignmodel/dense_w(model/dense_w/Initializer/random_uniform*
validate_shape(* 
_class
loc:@model/dense_w*
use_locking(*
T0*
_output_shapes

:w
x
model/dense_w/readIdentitymodel/dense_w* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:w
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
 *W�*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *W�;*
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
valueB�w*    *'
_output_shapes
:�w
�
model/VariableVariable*
dtype0*
shape:�w*
	container *
shared_name *'
_output_shapes
:�w
�
model/Variable/AssignAssignmodel/Variablemodel/zeros_4*
validate_shape(*!
_class
loc:@model/Variable*
use_locking(*
T0*'
_output_shapes
:�w
�
model/Variable/readIdentitymodel/Variable*!
_class
loc:@model/Variable*
T0*'
_output_shapes
:�w
�
model/Assign/value/0Pack0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�w*
T0*
N
�
model/Assign/value/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�w*
T0*
N
�
model/Assign/valuePackmodel/Assign/value/0model/Assign/value/1*'
_output_shapes
:�w*
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
:�w
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
:	�w
�
*model/gradients/model/MatMul_grad/MatMul_1MatMulmodel/Reshape&model/gradients/model/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0*
_output_shapes

:w
d
(model/gradients/model/Reshape_grad/ShapeShapemodel/concat*
T0*
_output_shapes
:
�
*model/gradients/model/Reshape_grad/ReshapeReshape(model/gradients/model/MatMul_grad/MatMul(model/gradients/model/Reshape_grad/Shape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell1/dropout/mul*model/gradients/model/Reshape_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell1/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�w
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
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
:���������w
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/mul_1Mulmodel/zeros_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�w
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�w
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
:	�w
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�w
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
:	�w
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
��*
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
��
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
:�
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
��
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
:	�w
�
Umodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice_1SliceTmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul\model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ConcatOffset:1Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/ShapeN:1*
Index0*
T0*'
_output_shapes
:���������w
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
:	�w
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
:	�w
�
Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/mul_1Mul(model/RNN/MultiRNNCell/Cell0/dropout/mulSmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat_grad/Slice*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
�
Cmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/mul_1Mul0model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Gmodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_1_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/mul_1Mul1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Emodel/gradients/model/RNN/MultiRNNCell/Cell0/dropout/mul_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/SquareSquare1model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1N^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
�
Jmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/mulMulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/ReshapeJmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1_grad/sub*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/sub*
T0*
_output_shapes
:	�w
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul_1MulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2_grad/Reshape_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2_grad/mul*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
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
:���������w
�
Imodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/mul_1Mulmodel/zerosMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/mul_1Mul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2_grad/Reshape_1*
T0*
_output_shapes
:	�w
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
:	�w
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
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mulMul2model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/sub*
T0*
_output_shapes
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_grad/Reshape_1Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_grad/mul*
T0*
_output_shapes
:	�w
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
:	�w
�
Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mulMul4model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Mmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/sub*
T0*
_output_shapes
:	�w
�
Omodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul_1MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/ReshapeMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1_grad/mul*
T0*
_output_shapes
:	�w
�
Kmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/SquareSquare/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhP^model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1*
T0*
_output_shapes
:	�w
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
:	�w
�
Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/mulMulOmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1_grad/Reshape_1Hmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_grad/sub*
T0*
_output_shapes
:	�w
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
:	�w
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
��*
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
��
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
:�
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
��
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
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_2MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_1Identitymodel/clip_by_global_norm/mul_2*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_3MulVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_2Identitymodel/clip_by_global_norm/mul_3*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_4MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_3Identitymodel/clip_by_global_norm/mul_4*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_5Mul*model/gradients/model/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:w
�
6model/clip_by_global_norm/model/clip_by_global_norm/_4Identitymodel/clip_by_global_norm/mul_5*=
_class3
1/loc:@model/gradients/model/MatMul_grad/MatMul_1*
T0*
_output_shapes

:w
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
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_6Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
\
model/zeros_7Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
\
model/zeros_8Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
f
model/zeros_9Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_10Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_11Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_12Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
c
model/zeros_13Const*
dtype0*
valueBw*    *
_output_shapes

:w
�
model/model/dense_w/AdamVariable*
	container *
_output_shapes

:w*
dtype0*
shape
:w* 
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

:w
�
model/model/dense_w/Adam/readIdentitymodel/model/dense_w/Adam* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:w
c
model/zeros_14Const*
dtype0*
valueBw*    *
_output_shapes

:w
�
model/model/dense_w/Adam_1Variable*
	container *
_output_shapes

:w*
dtype0*
shape
:w* 
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

:w
�
model/model/dense_w/Adam_1/readIdentitymodel/model/dense_w/Adam_1* 
_class
loc:@model/dense_w*
T0*
_output_shapes

:w
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
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_2*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_3*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
)model/Adam/update_model/dense_w/ApplyAdam	ApplyAdammodel/dense_wmodel/model/dense_w/Adammodel/model/dense_w/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_4* 
_class
loc:@model/dense_w*
use_locking( *
T0*
_output_shapes

:w
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
:�w*
T0*
N
�
model/HistogramSummary/values/1Pack0model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_20model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�w*
T0*
N
�
model/HistogramSummary/valuesPackmodel/HistogramSummary/values/0model/HistogramSummary/values/1*'
_output_shapes
:�w*
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
value	B :w*
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
:���������w
R
model_1/pack_1/1Const*
dtype0*
value	B :w*
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
:���������w
R
model_1/pack_2/1Const*
dtype0*
value	B :w*
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
:���������w
R
model_1/pack_3/1Const*
dtype0*
value	B :w*
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
:���������w
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
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�w:	�w:	�w:	�w
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
:	�w
�
4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�w
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mulMulmodel_1/zeros4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�w
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split*
T0*
_output_shapes
:	�w
�
1model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�w
�
3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�w
�
6model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�w
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
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
,:	�w:	�w:	�w:	�w
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
:	�w
�
4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/SigmoidSigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_1*
T0*
_output_shapes
:	�w
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mulMulmodel_1/zeros_24model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid*
T0*
_output_shapes
:	�w
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_1Sigmoid2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split*
T0*
_output_shapes
:	�w
�
1model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/TanhTanh4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:1*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1Mul6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_11model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2Add0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_1*
T0*
_output_shapes
:	�w
�
3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_1Tanh2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2*
T0*
_output_shapes
:	�w
�
6model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2Sigmoid4model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/split:3*
T0*
_output_shapes
:	�w
�
2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2Mul3model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Tanh_16model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Sigmoid_2*
T0*
_output_shapes
:	�w
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
:	�w
f
model_1/Reshape/shapeConst*
dtype0*
valueB"����w   *
_output_shapes
:
k
model_1/ReshapeReshapemodel_1/concatmodel_1/Reshape/shape*
T0*
_output_shapes
:	�w
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
valueB�w*    *'
_output_shapes
:�w
�
model_1/VariableVariable*
dtype0*
shape:�w*
	container *
shared_name *'
_output_shapes
:�w
�
model_1/Variable/AssignAssignmodel_1/Variablemodel_1/zeros_4*
validate_shape(*#
_class
loc:@model_1/Variable*
use_locking(*
T0*'
_output_shapes
:�w
�
model_1/Variable/readIdentitymodel_1/Variable*#
_class
loc:@model_1/Variable*
T0*'
_output_shapes
:�w
�
model_1/Assign/value/0Pack2model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2*#
_output_shapes
:�w*
T0*
N
�
model_1/Assign/value/1Pack2model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_22model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2*#
_output_shapes
:�w*
T0*
N
�
model_1/Assign/valuePackmodel_1/Assign/value/0model_1/Assign/value/1*'
_output_shapes
:�w*
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
:�w"	.�RL       �T�	�pr���A*�@

mean squared error[�C=

	r-squared HM:
�.
states*�.	   �v���   ����?    ��=A!饋C�?�@)����X�@2�3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W��T�L<��u��6
��K���7���i����v��H5�8�t�BvŐ�r�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�4�j�6Z�Fixі�W���x��U�H��'ϱS���Ő�;F��`�}6D��z��6��so쩾4�4�j�6Z>��u}��\>ڿ�ɓ�i>=�.^ol>ہkVl�p>BvŐ�r>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�������:�               @      @      M@     �f@      ~@     X�@      �@     ��@     �@     Y�@     �@     ѻ@     i�@    ��@    �]�@    ���@     ��@     ��@     ��@     ��@     2�@     u�@     �@    ���@     ��@     ��@     =�@     ��@    �"�@    ��@    ���@    �1�@    ���@     ��@     ��@     A�@     ,�@    ��@     0�@     ��@     �@     ��@     �@     \�@     ��@     ��@     ��@    ���@    @�@     ;�@    @��@    ���@    ���@     :�@    ���@    ���@    �4�@    @p�@    ���@    �?�@    �K�@    �b�@    �b�@     ��@     ��@    ���@    ���@     ��@     ��@     ~�@     �@     ��@     ��@     q�@     �@     ڲ@     H�@     ��@     Ƭ@     ,�@     P�@     ��@     ��@     ��@     f�@     ܟ@     |�@     ��@     ��@     |�@      �@     8�@     |�@     �@     �@     ��@     ��@     ��@     (�@     ��@     �@      |@     @y@     �u@     Pu@     0s@     �r@     q@     �m@     �g@     �g@     �f@     �b@     �f@     �c@     �]@      Y@      W@     �W@     �V@     �V@     @P@      K@     �P@     �E@     �E@     �M@      F@     �C@      4@      B@      9@      2@      ;@      &@      1@      3@      1@      .@      (@      @      @      $@       @       @      "@      @       @      @      @      @      @      @      @      @      �?       @       @              @              @      �?      �?              �?      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      @              �?              �?       @      @       @      �?       @       @      �?      @       @       @      �?      @      @      @      @      @      @      @      @      @       @      "@      (@       @      "@      *@      0@      $@      (@      2@      2@      1@      3@      9@      ;@      >@      <@      @@      9@      I@     �C@     �J@      I@      R@      Q@     �N@     �R@     �U@     @Z@     �X@      `@     @_@     �a@      d@     �g@     �i@      i@     �n@     �p@      r@     t@     0w@     �w@      z@     ~@     x�@     P�@     P�@     ��@      �@     ��@     `�@     ��@     đ@     0�@     @�@     H�@     ��@      �@     D�@     �@     �@     �@     ��@     ��@     T�@     ��@     ��@     I�@     r�@     ��@     ��@     B�@     º@     Լ@     ��@    ���@    �t�@    ���@     ��@    ���@     ��@    ���@     ��@     ��@    �a�@     D�@    ��@    �)�@    ���@    ���@    @��@    @�@    @��@     ��@    �1�@    @X�@     �@    ��@    ���@    ���@    �>�@    �R�@     ��@    ���@    ���@    ���@    �#�@     ��@     ��@    �]�@    ���@     L�@     ��@    ���@     ��@    @��@    �a�@     ��@    ���@    ���@    ���@    ���@    �F�@    ���@    ���@    ���@    �?�@     ��@    ���@    ���@    �C�@     >�@     S�@    �g�@     ܽ@     �@     3�@     <�@     ��@     �@     �@     ��@      v@      c@     �E@       @      �?        
�
predictions*�	   �hQ�   �WL�?     ί@! � )@)�Z���?2��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a���(��澢f���侮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پjqs&\�ѾK+�E��ϾT�L<��u��6
���*��ڽ>�[�=�k�>��~]�[�>��>M|K�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?�������:�              �?      �?      �?      �?       @      �?      @      @      @      "@      @      &@       @      "@      $@      @      @      $@      $@      "@       @      @      @      @      @      @               @       @       @       @       @      @      @       @       @       @      �?      �?              �?       @      �?      �?              �?              �?              �?              �?              �?               @              �?              �?      �?      �?              �?               @       @       @              �?              @      �?       @      @       @      @      @      @      @      @      @      @      @      @      @      "@      @      $@      "@      $@      @      "@      $@      (@      6@      6@      2@      5@      &@      5@      4@      :@      B@      3@      A@     �C@     �A@     �K@     �S@     �X@     @\@      `@     �b@      f@     �c@     �^@     `b@      f@     �h@     �l@     @m@     �o@     0r@      o@     �h@     �f@      X@     �L@      6@      @      �?      �?        }<mr      Bp�S	B_2r���A*�>

mean squared error��C=

	r-squared �$;
�,
states*�,	   �� �   �i;@    ��=A!T3���k�@)�����@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��39W$:���.��fc���X$�z��
�}�����4[_>�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�K���7�>u��6
�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�               @      �?      @      2@     �J@     �`@     @q@     �@     ȉ@     �@     |�@     ~�@     "�@     ��@     �@     ߷@     �@     �@      �@     ��@     e�@    �`�@    �B�@     ��@    ���@     ��@     Q�@    ��@     ��@    @�@     ��@    ���@    @W�@     ��@    @��@    ��@    ���@    ��@     �@     S�@    @��@    @�@    @=�@    @_�@    ��@    @d�@    @*�@     8�@    ���@    @�@    ���@     �@    ���@    ���@    �a�@     ��@     ��@     f�@    ��@     p�@     ��@     ��@     ��@     ��@     ޵@     �@     A�@     ��@     ƭ@     >�@     ��@     ~�@     ��@     f�@     .�@     ġ@     R�@     l�@     8�@     ��@     �@     `�@     (�@     ��@     L�@     �@     ��@     ��@     @�@     �@     ��@     H�@      �@     P@     �|@     pz@     �v@     �t@     �r@     Pq@     `o@     �m@      j@     @i@      d@     �c@     `b@     �`@      `@      \@     �U@     @U@      Y@     @R@     �V@     @P@     �J@     �J@     �F@     �M@      B@     �F@     �@@      8@      ?@      :@      3@      9@      7@      &@      5@      1@      $@      "@      $@      &@       @      "@      @      @      @       @      @       @      @      @       @      �?      @       @      @              @      @               @      @      �?       @               @               @              �?      �?              �?       @              �?      �?               @              �?      �?               @              �?      �?      �?       @              �?      �?              @       @       @      �?              �?      @      @       @       @      @               @       @      �?      @      @      @      @      @       @      @      "@      @      .@      .@      "@      ,@      .@      *@      8@      6@      ;@      ?@      8@      <@      >@      :@      H@     �D@     �F@     �K@      L@     �Q@      P@     �R@     @S@     @X@     @V@     �[@     �\@     �Z@     �`@     �a@      f@     �d@     @g@      j@      m@     �p@     �q@      s@      v@     `x@     Pz@     �}@     h�@      �@     ��@     ��@     ��@     P�@     �@     P�@     �@     \�@     �@     ��@     X�@     ؜@     ��@     J�@     ��@     ؤ@     6�@     ��@     �@     v�@     ȯ@     W�@     ��@     ��@     g�@     �@     ��@     ��@     h�@     Z�@    ���@     @�@    � �@    �n�@     ��@    �f�@     ��@    ���@    �A�@    ���@    ���@     7�@     �@    �S�@     �@    @0�@     ��@    �
�@    �Y�@    �t�@     D�@     �@    ���@     i�@    ���@    ���@    @��@    ���@    ���@    @�@    ���@    ���@    @E�@    ��@     ��@    @��@     ��@     G�@     �@     3�@     �@     ��@    ���@    ���@     ��@     ��@     ��@     )�@     ¼@     ��@     ��@     v�@     D�@     ��@     ��@     `�@     �v@     �e@      S@     �@@       @      �?      �?        
�
predictions*�	    ����   �x��?     ί@! `�f��P@)��D��?2����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7��������6�]���>�?�s���O�ʗ���T�L<��u��6
��K+�E���>jqs&\��>f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�������:�              �?              �?      �?      @      �?      @      $@      @      @      &@      @      &@      "@       @      @      &@      @      $@      @      @      &@      &@      $@      @      @      @       @      @      @      @       @      �?      �?      @      @       @      �?      @      @      �?      �?      @      @               @      �?      �?              �?      �?      �?      �?       @      �?      �?              �?               @              �?              �?              �?       @              �?      �?      �?               @      �?      @              @              @              �?      �?       @      @      @      @      @      @      @      @       @       @       @      �?      @      @      $@      @      $@      .@      $@      .@      &@      &@      3@      5@      1@      7@     �@@     �C@      I@      M@     �Q@     �U@      W@     @X@     �[@      `@     @b@      e@      e@     �j@      m@     �o@     �o@     �o@     `o@     �i@     �h@     �c@      _@     @U@      K@     �B@      3@      "@       @       @        fеl�       ����	:`r���A*�A

mean squared error	�@=

	r-squared�]�<
�+
states*�+	   ��^��   @�T @    ��=A!J͆�/ �@)d%GFY��@2�S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���
�}�����4[_>����
�%W����ӤP���K���7��[#=�؏�������~�f^��`{�������M�6��>?�J�����W_>�p
T~�;�u��6
�>T�L<�>��z!�?�>��ӤP��>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�               @      @      0@     �M@     �b@     �s@      �@     (�@     8�@     \�@     ��@     Բ@     ��@     �@    ���@    ��@     ��@    ���@    �F�@    � �@     6�@    �`�@    ���@     ��@     ��@    ���@    ���@    @5�@    ���@     ��@    �~�@    ���@    @~�@     A�@    �n�@     ?�@    ���@    ���@    ���@     ��@    � �@    ���@    ���@    @d�@    ���@    @&�@    ���@    @��@    �v�@     q�@    ���@     ��@    ���@     ��@     ��@     "�@    ���@    �9�@     /�@    ���@     ��@     Ǿ@     �@     ��@     M�@     ��@     ��@     ֱ@     ǰ@     ֭@     ��@     �@     �@     �@     �@     ��@     ؟@     ��@     @�@     P�@     ��@     Г@     p�@     L�@      �@     ��@      �@     ��@     8�@     ��@      ~@     P~@     �{@     0|@     v@     �s@     0s@     Pp@     �p@     �n@     �i@     @f@     �b@     @c@     `a@      `@     @]@     �X@      U@     �S@     �U@      R@      L@     �P@     �M@      H@      O@      G@     �@@     �A@      @@      @@      :@      A@      .@      1@      7@      1@      (@      3@      1@      "@       @      "@      $@       @      &@      @       @      @      @       @      @      @      @      @      @       @       @      @      @      �?               @       @      �?      �?      �?      �?              @      �?              �?              �?              �?              �?              �?              �?              �?      �?      �?              @       @      �?       @               @      @       @       @      �?       @      �?       @      @      @      �?      �?              @      @      @      @       @      @      �?      @      "@      @       @      @      "@      $@      @      4@      1@      3@      $@      (@      3@      :@      7@     �C@     �@@      ?@      C@      C@      D@     �D@     �I@     �M@     �Q@     @T@      X@     �V@     �W@      V@     �\@     �_@     �b@     @c@     �e@     `h@      h@     �l@     `k@     �q@     �r@     ps@      v@     �w@      }@     0|@     P�@     ��@     P�@     Є@     ��@     ��@     �@     �@     ��@     ��@     ��@     ��@     �@     T�@     v�@     �@     (�@     ��@     d�@     ��@     ��@     �@     �@     ��@     g�@     ��@     a�@     �@     G�@     ѿ@    ���@     ��@    ���@    �V�@     ��@    ���@    ���@    ���@     ��@    @��@    ���@    ��@    @��@    ���@    ��@    ���@    ��@    �+�@    ���@    @��@    @f�@    @��@    �^�@    @��@    @8�@    @��@    ��@    �K�@     ��@     ��@     ��@    @,�@     �@    @A�@     ��@    @��@    ���@    ���@    ��@    ���@    �`�@     #�@    ���@     =�@    ���@     N�@    ���@    ���@     �@     ø@     V�@     �@     T�@     ��@     x�@     P�@     �t@     �`@     @Q@      =@       @      @       @        
�
predictions*�	   �����   �tW�?     ί@! �J��&@)42��G�?2�
I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[����Zr[v��I��P=����~]�[Ӿjqs&\�Ѿ['�?�;;�"�qʾ�_�T�l�>�iD*L��>})�l a�>pz�w�7�>����?f�ʜ�7
?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?�������:�
               @      @      "@      5@      1@      C@     �J@     �M@      T@     @Q@     @X@     �W@     �V@      T@     @U@      T@     @S@     �P@      N@     �Q@     �O@      P@      I@     �G@     �G@     �G@     �A@      A@     �A@      @@     �B@      >@      3@      9@      3@      1@      &@      5@      (@      ,@      "@      $@      @      &@      "@      @      @      @      @      @       @      @      @      @      @      �?      �?      �?       @      �?       @      @      �?      �?              @      @      �?              �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?               @      �?       @              �?              �?              @       @       @              �?       @       @      @      @      @      @      @      @      @      $@      @      @      "@      @      0@      0@      &@      ,@      5@      ,@      0@      ;@      .@      .@      7@      >@     �@@      F@     �G@     �G@     �I@      E@      I@     �P@     �H@      G@     �R@      T@      N@     �Q@     @T@      Q@     @Z@      T@     �X@      V@      U@     �S@     �Q@     �P@      O@      H@     �A@     �B@      9@      1@      $@      "@      �?      �?        =�aՒ       ��K	7�r���A*�A

mean squared error�}A=

	r-squared��[<
�+
states*�+	   �� �   ���@    ��=A!l��kl�@)��=��@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�������������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������.��fc���X$�z���
�%W����ӤP���u��6
��K���7��7'_��+/>_"s�$1>����W_>>p��Dp�@>w`f���n>ہkVl�p>[#=�؏�>K���7�>u��6
�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�               @      @      @      @      1@     @P@     �\@     w@     h�@     Π@     <�@     ̰@     w�@     	�@    �.�@    �{�@    ���@    ��@    ���@     -�@    � �@    ���@     ��@     J�@    ���@     ��@     ��@    @"�@    ��@    �S�@    �;�@    @��@     ��@    ��@    ���@    �}�@    ���@    ���@     ��@    �L�@    �N�@    @��@    � �@    ��@    @��@    ���@    @�@     ��@    ���@    @��@    �&�@    ���@     ��@     ��@     ��@    �J�@     0�@    ���@    �%�@     ��@     �@     t�@     ��@     ��@     ��@     ۲@     *�@     �@     r�@     ��@     ��@     �@     ��@     2�@     .�@     �@     p�@     l�@     P�@     \�@     t�@     ��@     ��@     �@     ��@     P�@     ��@     ��@     ��@     h�@      |@     �y@     `w@     `t@     �r@     �p@     pr@      m@     �i@     �f@     �d@     `c@     @b@     �`@     �[@     �\@     �X@     �U@      V@     �T@     �R@      O@     �G@     �I@     �F@     �E@     �E@     �D@      @@     �@@      5@      7@      2@      4@      ,@      &@      8@      ,@      @      *@      $@      "@       @      @       @      @      @      @      @       @      @      @      @      �?      @      �?      �?      @      @      @       @      �?              �?               @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?       @      �?      �?              �?      �?      @       @       @              �?      �?      @       @              �?      �?       @      @       @      @      @      @      @      @      @      "@       @      *@      &@       @      .@      0@      &@      (@      1@      ,@      1@      5@      4@      8@      ;@      ;@      @@      8@      ?@     �B@      I@      P@      M@      O@     �Q@     �S@      V@      X@      ^@     �\@     ``@     �a@     �c@     �e@     @g@     �i@     �m@     �n@     �p@     �s@     �u@     �v@     �y@      |@     �@     ��@     H�@     �@     0�@     p�@     X�@      �@     �@     �@     x�@     X�@     h�@     4�@     <�@     ��@     �@     ��@     $�@     ާ@     Ҫ@     Ƭ@     ��@     �@     Ų@     ��@     ��@     ��@     /�@     ��@    ���@    ���@     ��@    ���@    ���@     ~�@    �x�@    ��@    �O�@    ���@    ��@    �G�@    �>�@    �-�@    ��@    ���@    ���@     ��@    �b�@    ���@     7�@    @��@    ��@    ��@    ���@    @{�@    @��@    ��@    @�@    �0�@     ��@     �@    ���@    ���@    @`�@    ���@     ]�@     �@    �R�@     ��@     ��@    ���@     ��@     ��@     ^�@     ��@     $�@     ��@     �@     
�@     ��@     p�@     x�@     �r@      b@     �L@      ;@      &@      "@      @      �?        
�
predictions*�	    ����    b��?     ί@!  ���2�?)7ce���@2�
!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����[���FF�G �})�l a��ߊ4F��h���`�E��a�Wܾ�iD*L�پI��P=�>��Zr[v�>6�]��?����?f�ʜ�7
?�T7��?�vV�R9?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�������:�
              �?      �?      @       @      "@     �A@      K@     @W@      U@      X@     @Y@     @\@     @Z@     �Z@     �X@     �W@     �Y@      V@     �X@     �T@      M@      P@     �K@     �I@     @Q@      O@      >@      E@      E@      D@      9@      5@      A@      >@      0@      8@      (@      5@      "@      $@      "@      &@      @      *@       @      $@      @      @      @       @      @      @       @       @      @      �?      @      �?      @              �?      �?              �?              �?       @      @       @      �?               @              �?              �?      �?              �?              �?              �?      �?              �?               @      �?              �?              �?       @      @      �?      �?      @      �?       @      @      �?      @       @               @      @      @      @      @      @      "@      @      "@      @      "@      (@      2@      "@      &@      2@      2@      4@      2@      6@      9@      ;@      4@      =@      =@     �B@      M@     �E@      H@     �H@     @P@     �F@     �F@     �D@     �Q@     �O@      P@     �R@     @S@     @Q@     �T@      O@      U@      K@      R@     �N@      F@      F@      E@      F@      2@      ,@       @      @       @      �?        �.�WB       �\�	(��r���A*�@

mean squared errorB�?=

	r-squared�l�<
�+
states*�+	   �3�   ��@    ��=A!�����N�@)d��c�^�@2���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ���|�~���MZ��K��R%������39W$:���.��fc����
�%W����ӤP�����z!�?��T�L<��u��6
��f^��`{�E'�/��x�cR�k�e������0c�[#=�؏�>K���7�>���m!#�>�4[_>��>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              @      @      @      *@      7@     �M@     @c@     @y@     x�@     ��@     �@     ��@     ��@     "�@     k�@     ��@     ��@    ���@     ��@     ��@    �E�@    �#�@    ���@    �&�@    �E�@    ���@    �i�@     ��@    ���@    ���@     "�@     ��@    �q�@    @��@    @�@    �d�@     ��@    ��@    � �@    ���@     ��@     ��@     U�@    ���@    @��@    ���@    @�@    @��@    ���@    ���@    ���@     ��@    ���@     ��@    ���@    �T�@    ���@     ��@     ��@    ���@    �{�@     �@     w�@     ��@     ��@     ܴ@     γ@     �@     ˰@     b�@     4�@     B�@     ��@     ~�@     Ԣ@     `�@     �@     �@     \�@     P�@     ̔@     x�@     �@     ��@     @�@     (�@     P�@     X�@     x�@     ��@      @     0�@     pz@     �x@     pw@     �t@     `r@     �q@     `n@     @j@     �i@     �g@     @f@     �a@     @_@     @_@     �[@      [@     �Y@      U@     @U@     @Q@      O@      R@     �I@      L@     �F@      K@     �@@      <@      F@      2@     �C@      7@      :@      .@      .@      4@      $@      (@      &@      0@      @      @      @       @      @      @      "@      @      @       @      @      @      @      @      @       @               @      �?      @       @       @      �?              @              �?      �?               @      �?              �?              �?              �?              �?              �?               @      �?              �?              �?       @               @       @       @      �?       @       @      �?       @      @       @      @      @      @      @              @      @      @      "@       @      *@      @      @      @      &@      *@      2@      *@      0@      ,@      2@      ,@      ;@      ;@      <@      ?@     �@@      B@     �A@     �H@     �F@     �J@     �N@     @S@     �P@      U@     �T@      ]@     �V@     @]@     ``@     @b@     �d@     �c@     �h@     �g@      l@      n@     q@     �t@     ps@     0w@     z@      |@      ~@     P�@     `�@     ��@     ��@     ��@     �@     ��@     �@     ��@     D�@     (�@     ��@     ��@     (�@     |�@     B�@     �@     �@     :�@     `�@     �@     ®@     ϰ@     ��@     ��@     ��@     G�@     &�@     ݼ@     �@     �@     ��@    ���@    ���@     �@     ;�@    ���@     ��@     +�@    @��@    @n�@     ��@    @d�@     ��@     7�@    @
�@    ���@    @��@    �@�@    @^�@    @Y�@     w�@    �,�@    �g�@    @��@     ��@    @%�@    @��@     ��@    ���@    @��@     �@    ���@    ���@    ��@    @_�@    ���@     ��@    ���@     �@     ��@     �@     ��@     ��@    ���@    ���@    �(�@    �,�@     W�@     ÷@     X�@     r�@     ��@     ��@     �@     @h@     �R@      C@      ,@      @      @       @        
�
predictions*�	   �&��   ��c�?     ί@!  "�4L;@)[�
�	@2�
����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��E��a�Wܾ�iD*L�پjqs&\��>��~]�[�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>��[�?1��a˲?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�
               @      @      @      2@      ;@      O@     �M@     �R@     @R@     �X@     �Y@     �V@     �S@     �T@     �U@     �N@     @S@     �P@      M@      M@      D@      I@      D@     �F@     �G@      B@      @@      >@      8@      3@      ,@      4@      3@      3@      .@      $@      (@      &@      @       @      @      @       @       @      @      @      @      @      @      @      @      @      @      �?      @       @       @       @       @      �?      �?       @              �?              �?      �?      �?              �?              �?              �?              �?               @              �?              �?              �?      �?       @       @      �?      �?       @               @      @      @      �?       @       @      @      @      �?      @      @      @      @      @       @      @      @      @      @      &@       @      "@      *@      .@      ,@      .@      "@      3@      0@      1@      :@      ;@      5@      8@      @@      =@      >@      G@     �N@      G@     �C@      L@     �P@     �Q@     @R@     �N@     �U@     �J@     @R@     @S@     �S@      N@     �S@     �Y@     �P@     �U@     @S@     @X@      P@     �S@      Q@      P@      K@      C@      1@      ,@      *@      @      �?      �?         Io�       ��K	���r���A*�A

mean squared error>v?=

	r-squared ��<
�+
states*�+	   ��4��    G @    ��=A!�:�>:��@)���25�@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�������������?�ګ�;9��R���5�L�����]�����u��gr��R%������39W$:���.��fc���u��6
��K���7��ڿ�ɓ�i>=�.^ol>BvŐ�r>�H5�8�t>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�               @       @      @      &@     �F@      c@     @z@     ԓ@     �@     ��@     �@     ��@     ˷@     ��@    ���@    �i�@    � �@     ��@     ��@    ���@    �)�@    ���@     2�@    �A�@    @e�@    @��@    ��@    ���@    @��@    @��@    @�@    @
�@    @��@    ��@    ��@    �a�@    �e�@     h�@    �;�@     {�@    @��@     7�@    �Y�@    �h�@    �H�@    ���@    @��@    ���@    @/�@    ���@    ���@    ���@    �n�@     @�@    �Q�@     ��@    ��@     k�@     ��@     ��@     ��@     ��@     J�@     �@     k�@      �@     �@     ��@     �@     Ы@     ��@     ��@     Ƥ@     D�@     T�@     ̠@     ��@      �@      �@     ȕ@     ��@     ��@     d�@     �@     P�@     P�@     ��@     h�@     @�@     Ȃ@     �@     �{@     �z@     �u@     0v@     t@     �p@     �p@     `j@     `j@      h@     `g@     `d@     �c@      `@      V@     �W@     �X@     @U@     �U@     @U@     �Q@      L@     �G@      G@      B@      F@      B@     �D@      8@      <@      5@      ,@      2@      2@      *@      0@       @      2@      @      @      $@      *@      "@      @      $@      @      @      @      @      @      @      @      �?      �?       @      �?      �?              @      �?              @      �?              �?               @              �?              �?              �?              �?              �?               @      �?              �?      �?              �?              �?      �?      �?      �?               @      �?       @      @              �?      �?               @      �?      @       @       @       @      @      @      @       @      @      @      @      @      $@       @      &@       @      $@      $@      &@      &@      @      (@      0@      7@      2@      :@      :@      3@      8@      @@     �D@      H@     �E@      P@     �N@      R@     @Q@     �U@     �V@      X@     @Y@     @Z@     �a@      d@     �a@     @b@     �e@      f@      l@     �n@      o@     pq@     �q@     s@     �u@     p{@     p{@     �{@     8�@     @�@      �@     `�@     ��@     ��@     ��@     ��@     �@     (�@     �@     �@     �@     |�@     ȟ@     z�@     �@     ��@     t�@     "�@     �@     ��@     ��@     �@     t�@     �@     �@     ��@     ��@     Ծ@     ��@    �R�@    ���@     U�@     �@     ��@    �"�@    ���@     ��@     ��@    ���@    @��@    �q�@    ���@    ���@    �4�@    ��@    @��@    @_�@    @��@     o�@    ��@     ��@    @��@    @��@     M�@     �@    �e�@    @��@    �8�@    ���@     ��@    @n�@     v�@    ��@     _�@    �U�@    @��@     �@     ��@     w�@     ��@     q�@    ���@    ��@    ��@    ���@     ��@     O�@     3�@     _�@     �@     d�@     ȑ@     �@      e@     �L@      .@      "@      @       @        
�
predictions*�	   �ש��   �[��?     ί@! @C4�5@)�L�ts@2�
I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1�>�?�s���O�ʗ���I��P=��pz�w�7�������>
�/eq
�>E��a�W�>�ѩ�-�>a�Ϭ(�>8K�ߝ�>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?�������:�
              �?      @       @      3@      @@     �I@      O@     @U@     @[@     �X@      Y@     �U@      R@     �Z@     �T@     @R@     @T@      R@      N@     �Q@     �E@     �F@     �I@      J@      A@      @@     �@@      @@      =@      6@      *@      5@      &@      .@      2@      *@      $@      &@      &@      *@      &@      $@      .@      @      "@      @      @      �?      @      @      @      @      @      @      @      �?      �?       @      �?       @      �?      @      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?       @      �?      @       @      �?      @              �?       @      �?      @      @      @      @      @      @      �?      @      @      @      @      @      @      @       @      "@      *@      (@       @      *@      &@      0@      2@      :@      8@      6@      8@      ?@      B@     �C@     �A@      I@     �H@     �Q@      J@      L@      Q@     �K@      I@     �S@      K@     @P@     �Q@     �R@     @Q@     �Q@     �R@      R@      O@      Z@     @S@     �S@     @Q@     �N@     �J@     �G@     �D@      =@      4@      $@      @      @       @              �?        "�H%       �0u9	�s���A*�?

mean squared error�>=

	r-squared Y�<
�*
states*�*	    ����   �} @    ��=A!-�=,Y�@)�iM+N��@2�S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R�����]������|�~���u��gr��R%������T�L<��u��6
��ہkVl�p�w`f���n�d�V�_���u}��\�28���FP�������M�f^��`{>�����~>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>.��fc��>39W$:��>�u��gr�>�MZ��K�>��|�~�>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @      $@      $@      M@     �c@     �y@     l�@     ��@     �@     ��@     ��@      �@    ���@     c�@    ���@     C�@     (�@     c�@     �@    �H�@    ��@    ���@    @��@    �d�@    @#�@     ~�@    @s�@     L�@     >�@    �a�@    @��@     `�@    ���@    ���@    ��@     i�@    ���@    @��@     ��@     h�@     ��@     ��@     5�@    �J�@    �@�@    ���@    ��@     #�@    ���@     ��@    ��@    �A�@    ���@    ���@     �@    ��@    �M�@    �$�@     ��@     ��@     	�@     ��@     #�@     g�@     ±@     �@     ­@     ��@     ��@     :�@     ��@     ~�@     ܠ@     \�@     ��@     ܙ@     P�@     |�@     `�@     А@     ��@     ��@     �@      �@     (�@     ��@     ��@     �@     ��@     `|@     �w@      x@     �u@     �s@     q@      q@     �n@     �j@     @i@     �d@     �a@     �a@     �`@      `@     �T@     @Y@     �R@     �P@     �O@      N@      P@      F@      I@     �I@     �A@     �B@      A@      @@      ;@      @@      3@      <@      9@      2@      4@      ,@      1@      (@      @      ,@      $@      $@      @      $@      @       @      @      @      @      @      @      @              @      �?      @      @              @              �?               @       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @              �?       @              �?      �?      �?      �?      �?       @      @      �?       @       @      @       @       @      @      @      @      @      @      @      @      @      @      @       @      @      @      "@      (@      .@      *@      2@      1@      1@      *@      5@      ?@      =@      B@      ?@     �B@     �F@      G@     �I@     @P@      M@     �Q@     �O@     �W@      W@     �W@     �Y@     �[@      _@     �]@      f@     �h@     �e@     �g@     �k@     pp@     �n@     �r@     @v@     Pv@     �x@      |@     �~@     `�@      �@     ��@     ��@     ؈@     0�@     ��@     0�@     �@     �@     ��@     ��@     \�@     8�@     ��@     ��@     أ@     ��@     ��@     <�@     ��@     �@     Ӱ@     ��@     ��@     �@     ɸ@     	�@     o�@     �@     ��@     q�@    ���@    ��@    ��@     G�@    �*�@     ��@    �R�@    �]�@    �^�@    �]�@    @��@     ��@    ���@     Y�@    �p�@    @�@     ��@    @	�@    @T�@    �z�@    ���@    �n�@    @��@    �j�@    @�@    @��@    �0�@     ��@    ��@    �~�@    @f�@    �p�@    ���@    �v�@    �7�@    @��@    ��@    ��@     ��@     �@     ��@     c�@    �Y�@    �q�@      �@     �@     �@     ,�@     ��@     <�@     Ȏ@     �v@     `c@     �P@      ?@       @      @      �?        
�
predictions*�	   `^���   @x�?     ί@! ��O�S0@)�ƃ�@2�
�{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r��h���`�8K�ߝ뾯��]���>�5�L�>�h���`�>�ߊ4F��>1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�������:�
              @      ,@     �D@      L@     �R@     @W@     @Z@      \@     @X@     �]@     �V@     �_@     @[@     �S@     �P@     @U@     �J@      Q@      P@     �N@     @Q@     �H@     �@@      ?@     �C@      >@      ?@      A@      5@      7@      9@      :@      7@      3@      6@      *@      @      ,@      0@      .@      "@      &@      @              @       @      @      @      $@      @      @      @       @       @      �?       @               @       @       @       @              �?       @              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      @       @      �?              @              �?       @      �?               @      �?       @       @      �?      @       @      @      @       @      �?       @      @      @      @       @      @      $@      &@      &@      .@      (@      .@      1@      7@      6@      5@      <@      @@      <@      7@      5@     �A@      J@      A@     �B@     �A@     �@@     �E@     �A@      E@      N@      P@     �K@     �N@     �O@      M@      K@     @P@     �P@     �O@      P@     @R@     �Q@     �J@     �L@     �N@     �H@      E@      ?@      :@      =@      &@      (@      @      @      @        ����      A�	@	2�Gs���A*�?

mean squared errorno==

	r-squaredЯ=
�)
states*�)	   ��M�   @�2@    ��=A!_�nv�@)8�6)�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ����?�ګ�;9��R���5�L�����]����f^��`{�E'�/��x�BvŐ�r�ہkVl�p�
�}���>X$�z�>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?       @      @      @      ,@     �G@     �]@      p@     ��@     ��@     Ƥ@     �@     z�@     x�@     ��@     ;�@    ���@     b�@     M�@     �@    �z�@    ��@    ���@    �^�@     ��@     	�@     ��@      �@    @e�@    ���@    ���@    �9�@     v�@     ��@    ��@    ���@    �|�@    ���@    ���@    @}�@    @��@     {�@     ��@    ��@    @4�@    �P�@    �5�@    @)�@    �c�@    @#�@    �Z�@     ��@    ���@    �R�@    ��@    ���@     �@    �_�@    �|�@     �@     5�@     "�@     f�@     ��@     �@     �@     ��@     K�@     ,�@     j�@     
�@     ڧ@     b�@     4�@     >�@     ؠ@     �@     �@     �@     0�@     ��@      �@     ��@     P�@     ��@     8�@     ��@     ��@     H�@     h�@     0@     �y@     0z@     �x@     �v@     t@     �r@     �p@     �i@     �h@     �j@      g@     �e@     �c@     �`@     @`@      X@     @U@      _@     �S@     @S@     �N@     �P@      G@      O@      N@      F@      A@     �@@      ?@      5@     �A@      2@      :@      <@      2@      ,@      ,@      (@       @      3@      .@       @      "@      @      @      @      @      @      @      �?      @      @      @               @      @      �?       @       @       @       @       @      �?              @              �?              �?              �?              �?              �?      �?              �?      �?      �?              �?       @      �?              �?      @       @       @      �?      @      @       @      @      @       @      "@      @      @      @      @      @      $@      *@      0@      "@      $@      ,@      .@      0@      6@      4@      A@      3@      <@      ?@      <@     �B@      B@      J@      G@     �F@      L@     �N@      O@     �R@     �Q@     @U@     @T@     @U@     @\@      _@     �`@     �b@      d@      d@     `e@     `l@     `o@     �o@     0r@     �s@     Pu@     �y@     pz@     �}@      ~@     0�@     (�@     x�@     ��@     ȉ@     X�@     �@     ��@     ��@     @�@     ��@     \�@     ě@     ��@     Π@     ~�@     �@     ܥ@     n�@     �@      �@     �@     ��@     �@     ��@     ж@     ��@     ��@     ��@    �Z�@     `�@    �H�@     ��@     f�@    �^�@     q�@    �8�@     ��@    ��@    �b�@     a�@    ���@    ��@    �O�@    @��@    @��@    ��@    @�@    �;�@      �@    @6�@     e�@     �@    @3�@    �s�@     ��@    @<�@     ��@     \�@    ��@    @��@    �?�@    ��@    @�@     5�@     ��@    @��@    @W�@    ��@     ��@     f�@    �%�@     �@     ~�@     j�@     d�@     _�@     B�@     5�@     v�@     ��@     ��@     `s@     @Z@     �G@      4@      @       @       @       @        
�
predictions*�	    ���   �
7�?     ί@! ��g��4�)��I� @2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9��>h�'��f�ʜ�7
�6�]���1��a˲���[���FF�G ��h���`�8K�ߝ�a�Ϭ(龮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�FF�G ?��[�?��d�r?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�������:�
              @      &@     �D@     �O@     �R@      \@     �`@     ``@      a@     @_@      b@      `@     �^@     �Y@     @Z@     �S@     �T@     �R@     �R@      T@     �R@      S@      J@     �G@      J@      D@      F@     �@@      B@      ?@      8@      9@      7@      =@      9@      9@      $@      3@      *@      (@      @      &@      "@      .@      @      &@      @      @      @       @      @      �?       @      �?       @      @       @      @              �?      �?               @      �?      �?              �?              �?              �?      �?       @              �?      �?              �?              �?              �?              �?       @              �?              �?      �?      �?      �?      �?               @      @       @      �?      @      @               @      @      @      �?      @       @      @      @      @      @      @      "@      @       @      @      @      @      0@      3@      "@      0@      6@      0@      ,@      6@      *@      :@      :@      6@      7@      9@      9@      :@      5@      @@      @@     �C@      E@      F@      G@     �J@      F@     �E@     �D@     �I@     �F@      J@      K@     �H@     �F@      F@      D@      C@      @@      E@      4@      0@      .@      *@      &@      &@      @      @      @       @      �?        T���!      Չ�~	�ss���A*�B

mean squared errorK�;=

	r-squared�J*=
�,
states*�,	   @� ��   `�5 @    ��=A!C&�N��@)�<�vC�@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z����ӤP�����z!�?��u��6
��K���7���H5�8�t�BvŐ�r�:�AC)8g>ڿ�ɓ�i>ہkVl�p>BvŐ�r>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>�
�%W�>���m!#�>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              @      $@      2@     �E@     @\@     0p@     ��@     D�@     �@     ��@     �@     T�@     �@    �/�@     p�@    ���@     ��@     ��@     g�@    �!�@    �:�@    ��@    @k�@    ���@     Q�@    ���@    ���@    ���@     6�@    �%�@    @��@    @��@    ��@    ���@     9�@    @�@    �Q�@    @a�@    ���@     V�@     ��@    ���@     ��@    ��@    �]�@     ��@    ���@    �B�@    @Y�@    �L�@     ��@     o�@     {�@     ��@    �W�@     ��@     <�@    ���@     ��@     <�@     �@     ��@     e�@     ��@     J�@     ��@     '�@     ��@     ,�@     ��@     Z�@     ʥ@     �@     ��@     ��@     ��@     ��@     �@     Ԗ@     ��@     x�@     ؑ@     ��@     `�@     �@     ��@     x�@     ��@     ��@     ��@     �z@     �{@     �w@     �v@     `s@     Ps@     0q@      l@     �i@      j@      g@      e@      d@     `b@     @a@      W@     �^@     @Y@     �W@     �Q@     �R@      N@     �K@      G@     �L@     �E@      F@      F@      @@      A@      >@      8@      (@      >@      3@      ,@      5@      .@      (@      @      &@      $@      (@      @      @      @      @      @      @      @      @      @      @      @              @      �?      �?      �?      @      @               @      �?      �?              �?      �?      �?      �?      �?      �?              �?               @              �?              �?              �?               @              �?      �?      �?      �?              �?              �?              �?              @              �?      �?              �?      �?      @      @      �?      @               @       @      @      @       @      @      @      @      @      @      "@      "@      @      @      "@       @      @      (@      @      0@      (@      .@      5@      3@      1@      ,@      :@      =@      ;@      H@     �F@      F@     �L@     �D@      N@      K@     @S@     �S@     @R@     �[@     �X@     @Z@     �\@     @`@     �b@     �d@     @e@     @h@     �j@     �o@      o@     �p@     �r@      s@     �u@     @y@     �|@     �~@     P@     H�@     �@     ��@     �@     ��@      �@     P�@     0�@     ��@     P�@     ��@     ��@     ��@     �@     f�@      �@     ��@     n�@     ��@     �@     (�@     :�@     �@     J�@     ޵@     �@     C�@     û@     ��@     ��@    ��@     ��@    ���@     ��@     �@    �z�@    �w�@     ��@     ��@     ��@     k�@    ���@    �n�@    @\�@    �Z�@     ��@    ���@    @R�@    @��@    @��@    ���@    @�@     ��@    @m�@     ��@    @��@    �U�@    ���@    �$�@     �@    @��@    @�@    @/�@     ��@    �_�@    ���@    �b�@    @��@    ���@    �8�@    ���@    �-�@     �@    �C�@    ���@    ��@    ��@     »@     S�@     �@     <�@     ��@     ��@     �s@     �\@     �I@      8@      "@       @      �?        
�
predictions*�	   ��9��   ����?     ί@! @�ߙ@)��>��@2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1�x?�x��>h�'����(��澢f����x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?�������:�
              �?      �?       @      A@     �C@      M@     �S@     �Y@      Y@     @X@      [@     ``@      \@     �Y@      W@      X@      X@      Y@     �V@     @R@      Q@      N@     �P@     �K@     �K@      I@      @@     �I@      A@     �@@      ?@      :@      5@      7@      3@      ;@      4@      4@      ,@      3@      $@      &@      @      @      "@      &@      @      @      @      @      @      @      @      @      @      @      @      @      �?      @       @      �?      �?      �?      �?              �?               @              �?              �?      �?              �?              �?              �?      �?              �?              @      @              �?       @              �?       @      �?      @       @      @      �?      �?       @      @      @      @      @      @      @      @      @       @      @      @      @       @      &@       @      @      "@      .@      3@      ,@      .@      2@      7@      :@      7@      4@      :@      =@      =@      A@     �@@     �F@     �B@     �F@     �F@      J@     �H@     �F@      E@      O@      H@      O@     @P@      I@      F@      C@      I@      M@     �I@      J@      G@      F@      <@      E@      ;@      4@      *@      0@      *@      2@      @      @      &@      @      @      �?      �?        baB"      �:V	;ǜs���A	*�D

mean squared error��;=

	r-squared ^,=
�,
states*�,	   �����    ��?    ��=A!꠨t���@)W�t���@2�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��X$�z��
�}�����4[_>������m!#���
�%W��T�L<��u��6
��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�=�.^ol�ڿ�ɓ�i�4�j�6Z�Fixі�W���x��U�/�p`B>�`�}6D>�
L�v�Q>H��'ϱS>u��6
�>T�L<�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?�������:�               @       @      7@     �P@     @d@     �q@     p�@     �@     ,�@     r�@     ��@     ��@     Q�@    ��@     ��@     l�@    �r�@    �#�@    ���@    ��@     �@    ���@    @-�@    @+�@    ���@    @��@    @8�@    ���@    @
�@    �t�@     ��@    @��@    @O�@    ���@     ��@    ���@    @�@     A�@    �V�@    @��@     ]�@    �$�@    ���@    @��@    ��@    @��@    @�@    �`�@    @��@    ���@    @��@     ��@    ���@    ���@    �R�@    �O�@     ��@     ��@    ��@    ���@     	�@     �@     Ϻ@     ��@     ��@     ��@     ��@     ��@     �@     Ҭ@     ��@     ��@     `�@     ��@     t�@     ,�@     ��@     �@     �@     ��@     ��@     L�@     ̑@     �@     8�@     ��@     (�@     ��@     ��@     8�@      �@     �}@     �{@     �x@     �w@     �v@     �s@     �q@     �l@     @m@     �k@      j@     �e@     �c@     �a@     @`@     �^@     �]@     @^@     �W@     �W@     �Q@     �P@     �L@     �K@      I@      J@     �H@      I@      @@     �B@      B@      9@      9@      4@      2@      6@      4@      $@      *@      0@      &@      0@       @      $@      "@      @      &@      "@      @      @      @      @      @      @               @      @      �?      @       @      @       @               @      �?      �?      �?      �?      �?               @               @      �?              �?              �?      �?      �?              �?               @              �?      �?              �?              �?              �?              �?              �?      �?      �?              �?       @              @       @      @      �?      �?      @      @       @      @      @      @      @       @      @       @       @      @      &@      @      $@      (@       @      "@      (@      $@      1@      (@       @      4@      1@      5@      :@      :@      @@      A@     �B@      G@     �B@     �C@      ;@     �O@      L@     �R@     �U@      T@     �U@      X@      [@      `@     �_@     @a@     �c@     �f@     �j@      i@     @m@     �o@     `s@     �q@      w@     �t@     �z@     �|@      ~@     P�@     ��@     �@     ��@     �@     ��@     ��@     ��@     p�@     ē@     `�@     ��@     �@     8�@     ��@     Ρ@     d�@     
�@     ��@     �@     ��@     ��@     "�@     ��@     �@     z�@     շ@     ��@     ؼ@     a�@     f�@    ���@     f�@    �Y�@    �c�@     d�@    �F�@     @�@    �{�@     ��@     ��@    ���@     ��@    ���@     ��@    @��@     ��@    @E�@    ���@     ��@     c�@    �~�@    @)�@     x�@    @��@     '�@    @{�@     ��@    �!�@    ���@    @>�@    ��@    �p�@    �8�@    @��@    @��@     P�@     M�@     ��@    ���@     C�@    ���@     ��@     ��@     ��@     �@     ��@    ���@    �n�@     ^�@     �@     ��@     ��@     $�@     x�@     u@      c@      I@     �@@      $@       @        
�
predictions*�	    ����   @1�?     ί@!  ���v@)����@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲�I��P=��pz�w�7���iD*L�پ�_�T�l׾��~]�[�>��>M|K�>���%�>�uE����>�f����>�h���`�>�ߊ4F��>��[�?1��a˲?6�]��?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?�������:�              �?       @       @      3@      C@     �F@      R@     �T@     �W@     �U@     �U@     �[@      Z@     �Y@     @Y@     �U@      Y@      ]@     �T@     �T@      Q@     @Q@      P@     �K@     �G@      D@     �E@     �B@     �F@      A@     �A@      6@      ;@      .@      .@      0@      1@      &@      2@      &@      ,@      &@      $@      $@      &@      @      @      @      @      @      @      �?       @      �?       @      @      �?      @       @       @      �?      �?      �?      �?              �?      �?       @      �?      �?              �?      �?      �?      �?      �?      �?      �?      �?              �?              �?              �?              �?              �?      �?               @              �?      �?              �?      �?              �?              �?      �?       @               @               @              �?      �?      @      @      �?       @      @      @      @      @      @      @      @      @      @      (@      @      $@      ,@      @      ,@      3@      :@      7@      3@      2@      9@      :@      =@      =@      <@      B@      >@      E@     �B@      G@     �C@     �H@      D@      F@      N@     �I@     �L@     �G@      N@      L@     �G@     �M@     �K@     �G@      K@      E@      E@     �H@      D@      B@     �I@      <@      A@      8@      6@      0@      4@      ,@      "@      &@      3@      @      "@      @      @       @      @      @        �|U�!      ��-�	���s���A
*�B

mean squared error	9=

	r-squared�h=
�,
states*�,	   ����   ���@    ��=A!&�����@)�* ��@�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:������m!#���
�%W����ӤP�����z!�?���H5�8�t�BvŐ�r�cR�k�e������0c�������M�6��>?�J��/�4��ݟ��uy�Fixі�W>4�j�6Z>T�L<�>��z!�?�>��ӤP��>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              @       @       @      $@      E@     @U@      j@     0{@     ��@     x�@     ��@     
�@     #�@     �@     D�@     �@     ��@     |�@    � �@     Z�@     �@    ���@    �`�@    �Z�@     ��@    �v�@    ���@    @z�@     ��@     ��@     /�@    �}�@     ��@    @e�@    ���@     B�@    @�@    ��@    ��@     X�@    @y�@    ���@    ���@    �L�@    ���@    @��@    ���@    @?�@    �y�@    ���@    @�@    �i�@    �Z�@     D�@     U�@    ���@    ���@     ��@    �U�@     ��@    ��@     �@     |�@     �@     q�@     ��@     o�@     z�@     ��@     �@     ��@     \�@     �@     �@     ��@     ȥ@     ��@     6�@     N�@     X�@     �@     (�@     �@     �@     ��@     ��@     ��@     Ћ@      �@      �@     ��@      �@     ��@     ��@     `~@     |@     0y@     �u@     @u@     �r@     Pp@      p@     `l@     �g@     �i@     �d@     �c@     �_@      a@     �^@      Y@     �P@      T@     �R@      U@      R@      P@     �L@      G@     �C@      =@     �G@      B@      @@      5@      <@      ?@      3@      8@      7@      6@      1@       @      3@      *@      &@      *@      (@       @      (@      @      (@      @      @      @      @       @      @      @      @      @       @      @      @      @       @       @       @       @      �?       @      �?       @      @              �?      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?       @      �?      @      �?      �?       @      �?      �?               @      @              @       @      �?      @      @      @      @      �?      @      @      @      @      @      *@      @      0@      @      (@      "@      @      ,@      2@      1@      4@      8@      4@      7@      6@     �B@     �C@      ?@     �E@      B@      J@      H@     �O@      L@     �S@     �T@      S@      W@      [@      [@     @\@     @_@     �d@     @e@     �b@     `i@     `i@     `o@     `o@     �q@     `s@     �t@      x@     `y@     0z@     �|@     ��@     ��@     0�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     Л@     ��@     �@     \�@     ��@     ��@     ��@     ̪@     �@     �@     B�@     Ӳ@     ȴ@     �@     +�@     }�@     �@    �\�@    ���@    �k�@    �(�@     ��@     _�@     8�@    �/�@    �\�@    �`�@    �g�@    @��@    ���@    @c�@     w�@    @A�@     ��@    ���@    ��@    ���@    @��@     ��@     ��@     >�@    ���@    ���@    �i�@     x�@    ���@     N�@    @��@    ��@    ��@     ~�@    @��@     ��@    �H�@    @*�@    ���@     ��@    ��@    �W�@    �#�@    ���@    �T�@     -�@     6�@     ~�@     h�@     ��@     ��@     5�@     Y�@     t�@     ơ@      �@     �u@     `d@     @R@      9@      .@      @      @      �?        
�
predictions*�	   `�#��    k��?     ί@!  �D��@)���C @2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��f�ʜ�7
������6�]����FF�G �>�?�s���R%������39W$:�����(���>a�Ϭ(�>f�ʜ�7
?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?ji6�9�?�S�F !?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?�������:�
              �?      @      6@     �G@      O@     @R@     �V@     @Z@      Y@     �X@      ^@     �V@     �[@      V@      S@     @W@     �T@     �Q@      P@      N@     �R@     �Q@     �M@      F@      D@     �B@     �E@     �F@     �B@      <@      4@      >@      5@      :@      1@      3@      5@      6@      .@      (@      (@      1@      @      @      @      @      @      @      @      @      @      @      @      �?      @      �?       @      @       @       @      �?      �?              �?      @      @              �?      �?               @              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @       @      @      �?      @              @      @      @      @      @      @      @      @      @       @       @      @       @      $@      @      ,@      .@      (@      &@      2@      *@      @@      =@      8@      <@      5@      @@     �@@     �B@     �A@      B@      F@      D@     �C@      D@      K@      M@      I@      I@     �F@      D@     �E@      F@     �M@      M@      H@     �N@      H@      M@     �J@      B@      F@     �B@     �B@      @@      =@      ;@      2@      3@      "@      .@      2@       @      @      @      &@      "@      @      @       @      �?       @        ��P�"#      ��Y	�qt���A*�F

mean squared error��:=

	r-squared02E=
�.
states*�.	    ��   @܆@    ��=A!K��̡@)x!�1��@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>�����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~��i����v��H5�8�t�ہkVl�p�w`f���n�:�AC)8g�cR�k�e��so쩾4�6NK��2���-�z�!�%�����`���nx6�X� �Fixі�W>4�j�6Z>E'�/��x>f^��`{>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              @      @      @      @      2@     �D@     @[@     �p@     P}@     @�@     �@     ��@     ��@     ��@     5�@     {�@     i�@     Q�@    �_�@     y�@     ��@    �r�@     l�@    ���@    ���@     ��@    @E�@    �C�@     ��@    �^�@    @��@     �@    ���@    �o�@     ��@    ���@    @'�@    �i�@    @4�@    ���@    @�@    ���@    �\�@    @�@    @��@    @;�@     f�@    ���@     /�@    �o�@    @y�@     V�@    ���@     ��@    ���@     ]�@     ��@     #�@     I�@     d�@     ��@     ��@    �e�@     1�@     ܽ@     ��@     ظ@     Զ@     ��@     1�@     e�@     ԯ@     ��@     *�@     �@     D�@     ��@     
�@      �@     X�@     ��@     ��@     ��@     D�@     ��@     ̑@     h�@     Ȍ@     0�@     X�@     �@     @�@     ��@      �@     P�@     �{@      {@     �y@     �t@      u@     �q@     q@     @l@     `k@     �f@     �f@     @g@     �b@     ``@      `@     �[@      \@     �S@     �P@      T@     �R@      T@      O@      L@     �E@      L@     �C@      G@     �D@      ?@      7@      >@      7@      @@      7@      5@      .@      *@      3@      @      0@      $@      $@      &@      *@      "@      @      @      @       @      @      @      @      @      @      �?      @      @              @      �?      @      �?      @      �?               @              �?      �?      �?              �?      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?               @              @              @       @      �?      @      �?       @      �?       @      @       @      @      @      @      @      @      @      @      @      "@      "@      0@      ,@       @      "@       @      .@      0@      *@      0@      5@      3@     �@@      :@      ;@      @@      :@      =@      F@      F@     �K@     �I@     �N@     �K@     �P@     @S@      W@     @X@     �[@     @[@      _@      `@     @b@     `f@     @g@     `i@     �n@     q@     @n@     �r@     pu@     �v@     �x@      y@     �}@     @~@     0�@     H�@     P�@     `�@     X�@     �@     �@     �@     �@     ��@     0�@     ��@     x�@     <�@     �@     V�@     ��@     ��@     ��@     .�@     ��@     ^�@     ��@     Y�@     p�@     z�@     ��@     |�@     h�@     ]�@    ���@     i�@    �!�@     �@    �"�@    ���@     |�@     ��@     ��@     ��@    ���@    ���@    @�@    �0�@    @�@    ���@     ��@     ��@     ��@    @6�@     '�@    @��@    ���@    ���@    �Z�@    �j�@    @n�@     L�@    ���@    �[�@    ��@    ���@     ;�@     �@     ��@    �r�@    ���@    �.�@    ���@     q�@      �@    ���@    ���@    ���@     ��@     ]�@     ��@    �L�@    �2�@     �@     U�@     o�@     H�@     ��@     Ȋ@     0v@     �e@      S@      C@      6@      $@      @      @       @        
�
predictions*�	    �n��   ����?     ί@!  5U-@)!뢃��'@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲�O�ʗ�����Zr[v��I��P=����(��澢f�����uE���⾮��%ᾋh���`�>�ߊ4F��>��Zr[v�>O�ʗ��>��[�?1��a˲?6�]��?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�������:�              @      @      <@     �H@     �S@     �S@     @^@     �^@     ``@     �Z@     @_@     �V@      W@     �U@     �U@     �M@      N@     @R@      L@     �I@      H@      F@     �J@      C@     �A@      D@      :@      <@      ?@      @@      8@      5@      9@      5@      3@      "@      "@      .@      @      @      $@      @      "@      @      "@       @      (@      @      @      @      @       @      @      �?      @      @       @      @              �?      �?       @      @       @      �?               @              �?              �?      �?              �?               @              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?      �?      �?      �?      �?      �?      �?       @      �?              �?      @       @      @       @      @      @       @      �?      @      @      @       @       @      @      $@      @       @      (@      "@      "@      ,@      ,@      "@      2@      4@      (@      4@      9@      :@     �@@      ;@     �B@     �C@     �D@      A@     �B@     �G@     �I@      E@      G@      G@      I@     �I@     @Q@     �I@     �J@     �G@     �Q@     �Q@      L@     �K@      K@     @P@     �K@     �H@      I@     �B@      <@      7@      ;@     �B@      <@      :@      1@      4@      ,@      @      (@      *@      @      @      @      @       @       @       @        ����$      ��ӿ	ڒ1t���A*�G

mean squared error]88=

	r-squared`�x=
�0
states*�0	   ��/�   @W@    ��=A!퐂QA���)1i�1���@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z���
�%W����ӤP�����z!�?��[#=�؏�������~�E'�/��x��i����v��H5�8�t�ہkVl�p�w`f���n�H��'ϱS��
L�v�Q�_"s�$1�7'_��+/���M�eӧ�y�訥��J>2!K�R�>������M>28���FP>�
L�v�Q>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>BvŐ�r>�H5�8�t>�i����v>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?      @      @      $@      7@      C@     �Y@     �o@     0~@     0�@     ��@     ƨ@     ��@     ��@     Ѻ@     �@    ���@    �
�@     ��@    ���@    ���@     ��@     J�@     l�@    �-�@     >�@    @��@    �5�@    @�@    ���@     ��@    @��@     ��@    ���@    �1�@    @��@    ���@     ��@    �W�@    @��@    ���@    ���@    �t�@    @O�@    �!�@     ��@    @t�@    �#�@    ���@    @��@     /�@    �Z�@    @�@     ��@    @6�@    ���@    ���@     ��@     ��@    ��@     F�@    ���@    �*�@     s�@     ��@     �@     {�@     q�@     ��@     ��@     ~�@     v�@     .�@     $�@     �@     ��@     @�@     ^�@     b�@     ��@     l�@     �@     @�@     ��@     ��@     p�@     �@     ��@     L�@     (�@     Љ@     ��@     �@     `�@     ��@     �@     �@     P{@     �x@     pv@     �u@     �r@      s@     `m@      k@     �l@      f@     �e@      i@     �b@     �a@     ``@     �\@      W@     @X@     @V@     �W@     @R@     �P@      N@     �J@      H@     �F@     �J@      ?@     �C@      <@      ;@      7@      ;@      8@      9@      1@      *@      *@      $@      *@      @      1@      $@      (@      "@      @      @      @      &@      @      @      @      "@      �?      @      @       @       @       @      @              @      @              �?      �?              @       @      �?              �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?              �?       @              �?              �?       @      �?               @      �?               @       @      @              �?              �?      �?       @      @      @      @       @      @      @      �?      @              @      @      @       @      @      @      @      "@      @      @      "@      $@      *@      @      .@      (@      *@      3@      (@      ,@      5@      2@     �@@      :@      ;@     �A@     �B@      F@     �D@      F@     �E@     �O@     �R@     @V@     @S@      V@      T@     �X@     �^@     �^@     �^@      e@     `c@     �e@     �j@     �i@     @n@      o@     �r@     @s@      v@     �u@     �|@     0z@     p}@     ؀@     ��@     @�@     �@     8�@     P�@     �@     @�@     ��@     ��@     X�@     ԕ@     �@     x�@     ��@     ��@     ��@     ڣ@     <�@     ��@     �@     �@     ��@     >�@     ��@     ��@     w�@     E�@     B�@     7�@     y�@    ���@     ��@     v�@     �@    ��@    ���@     ��@    ���@     �@    ���@    @��@    �J�@     �@     =�@    @��@    @��@    @��@    ���@    ��@    �K�@    �g�@    �c�@    @��@    �P�@    @W�@    ���@     ��@    �%�@    ��@     ��@    @S�@    ��@    �j�@     c�@     ��@    ���@    �^�@     ��@    ���@     Z�@     ��@     ��@    �n�@     T�@    ���@     ��@    �O�@    ���@    ���@     ƾ@     ¹@     h�@     L�@     ��@     ��@     P�@     Pv@     �a@     �K@      ?@      ,@      @      $@      @      �?        
�
predictions*�	   ���   ��=�?     ί@!  ��5	*@)����~�$@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��>�?�s���O�ʗ���})�l a��ߊ4F��f�����uE����jqs&\�ѾK+�E��Ͼ5�"�g���0�6�/n������?f�ʜ�7
?x?�x�?��d�r?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?\l�9�?+Se*8�?�������:�              �?       @      4@      @@      N@     �T@      X@      \@     �\@      W@      ]@      Z@      X@     �U@     @S@      W@     �P@      O@     �I@      K@     �N@      H@      L@      J@      @@     �D@      <@      C@      8@      8@      ;@      1@      5@      7@      4@      2@      *@      2@      1@      "@      .@      @      @      $@      *@      @      @      @      @      @      @      @      @              @      �?       @       @       @               @      �?       @      �?       @      �?      @      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?      �?      @               @      �?       @      �?               @              @      @       @              @              @      @      @      @      @      @       @      @      @      @      @      @      @      "@      *@      2@      *@      ,@      (@      7@      3@      7@      :@      @@     �A@     �@@      B@      E@     �F@      @@      =@     �G@     @P@      F@     �P@      M@      O@      P@     �K@     �P@     �N@      Q@     �L@     �K@     �L@      I@     @Q@     �F@      G@      F@      E@     �B@      B@      A@      ?@      .@      2@      9@      2@      &@      @      @      $@      @      @      @      @      @      �?       @       @              �?        t@I��#      �M	x	�*]t���A*�G

mean squared error87=

	r-squared��=
�0
states*�0	   �\t�   �5�@    ��=A!��7d���)5�\-&�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z�����m!#���
�%W����ӤP�����z!�?��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS���Ő�;F��`�}6D�6NK��2�_"s�$1���R����2!K�R���`���nx6�X� �nx6�X� >�`��>��o�kJ%>4��evk'>6��>?�J>������M>�
L�v�Q>H��'ϱS>d�V�_>w&���qa>�����0c>cR�k�e>ہkVl�p>BvŐ�r>�i����v>E'�/��x>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?      @      @       @      3@      D@      X@     �h@     �y@     H�@     ��@     ަ@     ��@     �@     J�@     ��@    ���@    �8�@    ���@    �_�@     ��@     8�@    �j�@     �@     ��@     ��@     J�@    @��@    ���@    ���@    @�@    �N�@    �c�@    @��@     ^�@    �"�@     a�@    @��@    �(�@    ���@    �N�@     D�@    ���@     $�@    ���@    @��@    ���@    �`�@    �]�@    @��@    �s�@    ���@    @(�@    �>�@    �m�@    �|�@     x�@     X�@     �@    ���@     ��@    ���@     k�@     ��@     h�@     ��@     ��@     �@     =�@     ��@     ��@     �@     o�@     f�@     n�@     \�@      �@     4�@     أ@     r�@     Р@     ��@     $�@     ��@     �@     ��@     t�@     ��@     @�@     �@     �@     ��@     ��@     `�@     H�@      �@     �}@     �}@     @{@     pw@     �w@     pr@     �r@     @q@      k@      j@     @k@     �g@     `d@     `e@     �b@      b@     �_@     �[@     @Z@     @T@     �W@     @W@     �I@     �L@     @P@     �M@     �J@     �I@      G@     �E@      ;@      :@      =@      3@      :@      6@      5@      4@      0@      &@      *@      (@      0@      &@      1@      $@      @      @      *@      "@       @      @       @      "@       @      @      $@      @      @       @       @      �?      @      @      @       @              �?               @       @              �?              �?              �?               @      �?      �?      �?              �?              �?              �?              �?      �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?       @      �?              @      @      �?       @      �?       @      @      @       @      @      @      �?      @      @       @       @       @      @      @      @      "@      $@      $@      @      ,@      @      @      (@      (@      4@      .@      *@      *@      0@      7@      8@      =@      C@      =@      @@     �A@     �@@     �K@      M@      =@     �N@      H@     �P@     �R@     �S@     @U@     �R@      W@     @W@     �^@      a@     �a@     �f@      d@     @h@     `h@     �h@      n@     �n@     r@     pr@     `u@     Pt@     �z@     �y@     H�@     ��@     x�@     ��@     8�@     ��@     (�@     ��@     4�@     D�@     `�@     ��@     �@     ��@     (�@     ��@     ��@      �@     ��@     �@     �@     T�@     t�@     ��@     x�@     ��@     ��@     ��@     �@     ��@     	�@     ��@    ���@    ���@    ���@    �.�@     1�@     0�@    �E�@     �@     T�@     ��@    ���@     ��@    @��@     ��@    ���@    ���@    ���@    �L�@    @��@    �H�@    @$�@    ���@    �;�@    �u�@    ���@    @��@     	�@    ���@    � �@    @��@     ��@     �@    ���@    @��@     p�@    �"�@    ���@     ��@     ��@    ���@     �@     ��@     �@    ���@    �R�@    �N�@      �@     B�@     ��@     >�@     L�@     ��@     p�@     B�@      �@     N�@     �@      l@      _@      L@      =@      .@      &@      @       @        
�
predictions*�	   �Q���   @ ��?     ί@!  �w8�?)��Z^!@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7���>�?�s���O�ʗ�����Zr[v��I��P=���uE���⾮��%ᾮ��%�>�uE����>a�Ϭ(�>8K�ߝ�>��Zr[v�>O�ʗ��>��d�r?�5�i}1?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?uo�p�?2g�G�A�?�������:�
               @      @      0@      @@     �O@     �P@      T@     @[@     �Y@      \@     @\@      Z@     @W@      Y@     @Y@      [@     �W@     @Y@     �V@     @Q@     �S@     �Q@     �G@      M@     �J@      J@      A@      A@     �D@      @@     �C@      ;@      8@      5@      .@      *@      ,@      $@      @      (@      *@      ,@      "@      *@      @       @      @      @      @      @       @       @      �?      �?      �?       @       @       @      @              @      @       @               @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?               @      �?       @      �?               @              �?       @      @      @       @       @      @       @      �?      �?      @      @       @       @       @       @       @      *@      .@      $@      0@      &@      $@      1@      7@      .@      <@      <@      @@      D@      >@     �E@      B@      >@      C@     �L@      H@     �K@     �M@     �I@      J@      J@      G@      H@      I@     �J@     �H@     �D@      G@      F@     �F@     �A@      E@      ;@     �D@      ?@      3@      6@      ;@      :@      2@      *@      *@      @      0@      @      *@      @      @      @       @      @       @       @      @              �?        %S���$      �j.�	�܈t���A*�I

mean squared error'i7=

	r-squared���=
�1
states*�1	   �n��   ��O@    ��=A!�JX_�4��)�����@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏��E'�/��x��i����v��H5�8�t�BvŐ�r�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W�H��'ϱS��
L�v�Q�6��>?�J���8"uH�_"s�$1�7'_��+/�;3���н��.4Nν�Bb�!澽5%�����'v�V,>7'_��+/>_"s�$1>��8"uH>6��>?�J>4�j�6Z>��u}��\>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?      @      �?       @      1@      >@      T@     �c@     0v@      �@     �@     ��@     r�@     =�@     �@     g�@     ��@    ���@    ���@    ���@     �@    �1�@    �S�@     {�@     Y�@    @��@    ���@    ���@     ��@    �&�@    �\�@    @��@    @��@     5�@    @��@    ���@     �@     9�@    @n�@    @N�@    ���@    @�@    ���@    @�@    �M�@    �!�@    @��@    @��@    �l�@    ��@     \�@    @��@    ��@    �6�@    @F�@    @R�@    ���@    ���@    ��@     �@    ���@    �A�@    �t�@    �M�@     #�@     ��@     ��@     ��@     -�@     ��@     d�@     ʲ@     N�@     �@     ��@     
�@     �@     x�@      �@     �@     B�@     ��@     ��@     �@     �@     Ԕ@     8�@     ,�@     p�@     �@     ��@     0�@     X�@     Ѓ@     x�@     ��@     �~@      }@     �y@     �x@     `v@     0s@     �p@     Pp@     �k@     �i@     �h@     �d@     �d@     �a@     @a@     @`@      ^@     @W@     @[@     @R@     �R@     �R@     �Q@      P@     �G@      M@     �G@      E@     �B@     �A@      6@      B@      A@      =@      B@      8@      8@      (@      6@      2@      .@      .@      2@      &@      $@      @      $@      (@      "@       @      $@      $@      @       @      @      @      @      @      @      @      @       @       @      @      @      @      @      @      @      �?       @      �?       @       @       @       @      �?      �?               @       @              �?              �?              �?      �?      �?              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?      �?       @      �?              �?      �?      �?       @      �?      �?              @              �?      @              �?              @       @       @              @      @      @      @       @       @      �?      @       @      $@      @      @       @       @      &@      @      (@      1@      (@      &@      0@      .@      0@      4@      <@      2@      4@      1@      3@      <@      ?@     �@@     �F@      @@     �D@     �D@     �K@      F@      G@     �O@     �O@     @Q@     @S@     �S@     @U@     @Z@     �^@     �^@     �_@     `a@     �a@     `b@     @i@     �i@     �k@      o@     �p@     `q@     �s@     0u@      x@     �w@     �{@     0�@     `~@      �@     ��@     �@     �@     `�@     Ќ@     ��@     \�@     |�@     <�@     �@     Й@     �@     �@     R�@     ܢ@     ��@     <�@     �@     ��@     ��@     ��@     +�@     ^�@     x�@     �@     �@     ��@     �@    ���@     ��@     ��@     ��@    �d�@     3�@     A�@     ��@    @�@    @��@     ��@    @f�@    �>�@    @z�@     &�@    �6�@    ���@    @�@    �3�@    �P�@    �~�@     �@    @�@    @e�@    ���@     �@     q�@    @-�@     ��@    @`�@    �G�@    �T�@    ���@    �m�@    @��@    ��@    ���@     ��@     ?�@    ���@     ��@     a�@     ��@    ���@     ��@    ���@    ��@     �@     ��@     P�@     �@     �@     Q�@     ��@     ��@     *�@     h�@     @j@     �]@     �K@      7@      $@      @      �?      �?        
�
predictions*�	   `kW��   �� �?     ί@!  ���a'@)H&ŻM(@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���vV�R9��T7����5�i}1���d�r�f�ʜ�7
�������FF�G �>�?�s����f�����uE���⾞[�=�k�>��~���>����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?�������:�              @      *@      A@     �Q@      W@     @Z@     �`@      ]@     �^@      \@      ]@      _@     �Y@      X@     �U@     @V@     �S@     �Q@     �L@      L@     �J@      D@      E@      ?@      ?@      <@     �C@      ;@     �A@      9@      <@      3@      3@      .@      0@      &@      .@      1@      (@      &@      @       @      @      @      @      @      @      @      @      @      @       @       @       @      @       @       @               @       @       @       @              �?              �?       @       @       @      �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?               @              �?      �?      �?      �?               @               @               @      �?      @      @      @      @       @      @      @      "@       @      @      @      @      @      @      *@      *@      @      @      $@      "@      (@      7@      1@      9@      7@      :@      :@     �@@      :@      @@      B@      @@      C@      E@     �K@     �@@     �A@      I@     �D@      J@     @R@      E@     �I@     �Q@      K@     �L@     �L@     �E@      H@     �H@      J@     �E@      H@     �@@     �B@      @@     �@@      6@      6@      2@      1@      1@      @      (@      $@      @      @      @       @      @      @      �?      @      �?        U,�rR$      �:	�¹t���A*�H

mean squared error��1=

	r-squaredX��=
�1
states*�1	   ����    �@    ��=A!khx��o��)2�� N�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP��������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�cR�k�e������0c�w&���qa�d�V�_�p
T~�;�u 5�9��z��6��'v�V,����<�)��1�ͥ��G-ֺ�І�5%���=�Bb�!�=�i
�k>%���>���<�)>�'v�V,>��Ő�;F>��8"uH>��x��U>Fixі�W>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�               @      @      @      (@      5@     �@@     @S@      a@     �r@     ��@     ��@      �@     ��@     ��@     *�@     ٽ@     �@     "�@     *�@    ���@    ���@      �@     �@     d�@     ��@    �
�@    ���@     ��@    @��@    ���@     ��@    ���@     ��@    ��@    ���@    �`�@    ���@    @��@     ��@     ��@     ��@    @T�@    ���@    @0�@    ���@    @��@    ���@     ��@    �N�@    @9�@    �i�@    @��@    ���@    �S�@    �7�@    �o�@    ���@     �@     h�@    �B�@    �m�@    ���@    ��@     ��@    ��@    ��@     ��@     պ@     �@     ��@     P�@     �@     ڰ@     Į@     �@     ��@     ܦ@     F�@     T�@     ޡ@      �@     ��@     ԛ@     ��@     ��@     `�@     t�@     �@     ��@     ��@     ��@     �@     ��@     ��@     X�@     `�@     �~@     �}@     y@     �y@     pu@     �s@     �p@     Pp@     �n@      j@     �i@     �f@     @d@      b@     �a@     ``@      `@      _@     �U@     �W@     �Y@     �R@     �S@      Q@      K@      J@     �J@      N@      E@     �C@     �@@      ;@      >@      8@      :@      8@      8@      8@      2@      .@      4@      0@      ,@      $@      (@      &@      $@      @      @      @       @      @      @      @      &@      @       @      @      @      @      @      @      @      @      �?      @      @      �?              �?      @      @       @      @       @      �?              @       @       @              �?              �?               @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?               @      �?      �?      �?      @              @              @      @      �?      @              @      @      �?      @      @      @      @      @      @      @      @      @      @       @      @       @      @      ,@      @      .@      @      (@      @      *@      2@      7@      0@      (@      .@      9@      3@      5@      2@      :@      5@      6@     �E@      >@      H@     �D@      C@     �B@      F@     �J@      O@     �O@     @Q@     @S@     @S@     @W@     �\@     �]@      `@      [@     �^@      d@      e@      g@     @h@      j@     �l@     @p@     �p@     `u@     t@     �x@     �w@     P~@     �@     @�@     ��@     8�@     ��@     �@     `�@     8�@     ��@     �@      �@     ��@     ��@     $�@     T�@     �@     |�@     h�@     �@     �@     ��@      �@     �@     	�@     p�@     �@     �@     �@     G�@     X�@     �@     ��@    �	�@     ��@    �g�@    ���@    ���@    ���@    ��@    ���@    @@�@    �)�@    ���@     ��@    @��@    �p�@    @��@     ��@    @��@    ��@    ���@    @t�@    ��@     p�@    ���@    ���@    ���@     �@     ��@    @��@    �E�@     ��@    �I�@     ��@     V�@    �q�@    ���@    �6�@     �@    �<�@     ��@    ���@     ��@     S�@    �l�@     �@     �@     ��@    �z�@    �o�@     ��@     Ļ@     �@     4�@     ®@     ��@     (�@     ��@     �m@      Y@      K@      7@      (@      *@      @       @        
�
predictions*�	   @����   ��_�?     ί@!  ����)���DS.@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�>�?�s���O�ʗ���f�ʜ�7
?>h�'�?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?cI���?�P�1���?�������:�              @      @      (@      C@     @R@     �Z@      ^@     @a@     @`@      ^@     �_@     �`@     @`@     @[@     @Z@      Y@      V@     @S@     @R@     �P@     �P@      L@      O@      I@      F@     �F@      D@     �@@      =@      @@      8@      >@      1@      3@      9@      ,@      *@      0@      *@      0@       @      @      @      @       @      (@      @      "@      @              @      @      @      @      @      @               @       @              �?       @              �?               @              �?              �?      �?              �?              �?      �?              �?              �?               @              �?      @      �?      �?              �?      �?      @              �?              �?              �?       @      �?      @       @       @      @      @      @      @      @      @      @      �?       @      &@      @      *@       @      $@      "@      "@      2@      3@      4@      2@      4@      1@      4@      ;@      ;@     �D@      ?@      5@     �H@      B@      D@     �H@     �F@      ;@     �F@     �@@      E@     �K@     �J@      F@     �K@     �I@     �D@      I@      D@      E@      F@     �C@      >@     �@@      :@      >@      9@      4@      ,@      8@      ,@      ,@      $@      &@      @      @       @      @      @       @              �?      @              �?      �?              �?        ��5��&      �
o	���t���A*�M

mean squared error3=

	r-squared���=
�6
states*�6	   ����   �]�@    ��=A!k�
_��)T���S��@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�d�V�_���u}��\���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F�/�p`B�p��Dp�@�7'_��+/��'v�V,�4��evk'���o�kJ%���R����2!K�R���J��#���j�_�H�}��������嚽        �-���q=:[D[Iu='1˅Jjw=V���Ұ�=y�訥=H�����=PæҭU�=2!K�R�>��R���>4��evk'>���<�)>�'v�V,>6NK��2>�so쩾4>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              @      @      &@      @      >@      @@     @Q@     �\@      n@     P�@     �@     ��@     ��@     �@     :�@     �@     ��@    ���@    �B�@     T�@    �5�@     �@     �@    �H�@     ��@    ���@    ���@     �@    ���@     ?�@    @a�@    @��@    @
�@    ��@    @/�@    �D�@    �Q�@    �V�@     ��@    ���@     ��@    �=�@    @��@    @��@    @�@    ���@    �^�@    @��@    ���@    �V�@    ���@    ���@     �@    �a�@    ��@    ���@    ���@    ���@     �@    ���@    �F�@    �<�@    �X�@     ��@    �*�@    ���@     ��@     ��@     J�@     ��@     ��@     �@     L�@     a�@     گ@     ��@     ��@     >�@     *�@     f�@     B�@     Ƞ@     ��@      �@     ș@     ��@     Ĕ@     ԓ@     (�@     ��@     �@     ��@     ��@      �@     X�@     p�@     P�@     �~@     �{@     �y@      w@     �v@     pq@     @s@     �o@     `m@     �m@     �i@     �g@     @f@     @c@     `c@      ]@     �`@     @[@     �Z@     �S@     �V@      R@      U@     �P@      O@     �O@     �F@     �J@     �H@      B@      B@      B@      G@      7@      @@      5@      >@      8@      0@      2@      *@      1@      ,@      &@      &@      @      @      *@       @      $@      $@       @      @      @       @       @      @       @      @      @      @      @      @      @       @      @      �?      @      @      �?      @      @      @      �?      �?      @       @      �?      �?      �?       @       @               @       @       @              �?      �?      �?              �?              �?      �?              �?       @              �?               @              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?               @      �?      �?              �?              �?       @       @               @              �?              �?      �?       @      �?      @      @      �?      �?              �?      �?      �?       @      �?       @       @      �?       @      @               @      @      �?               @      @       @      @      @      @      @      @       @      @      @      @      @      @      $@       @      @      "@      &@      @      *@      $@      2@      2@      8@      3@      1@      *@      .@      3@      9@      :@      ;@      B@      =@      E@      ?@      <@     �@@      J@     �G@      I@      L@      P@     �O@     �N@     �T@     �V@     �U@      V@     �X@      \@      _@     �`@     �^@      a@     @d@     �i@     �g@     �m@     �k@      m@     pp@     ps@      u@     �u@      y@     Pz@     p~@     �~@     ��@     @�@     �@     ��@     ��@     X�@     ��@     l�@     �@     �@     T�@     ��@     \�@     ��@     ��@     ��@     ��@     �@     ��@     .�@     �@     ��@     �@     ��@     }�@     +�@     �@     i�@     y�@     M�@     ��@    ���@     �@    �8�@    �B�@    ���@     m�@     ��@    �,�@    ���@    ���@    ��@     ��@    ���@     ��@    �r�@     ��@     c�@    �!�@    �p�@     _�@    ���@    ��@     e�@    @��@    ��@    �?�@    ���@    @��@    ���@    @��@    �w�@    @�@    �p�@    ��@     ��@     @�@    �:�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     I�@     �@    �-�@     U�@     {�@     ۾@     �@     ,�@     ٳ@     :�@     *�@      �@     ��@     �i@     @V@      D@      3@      (@      @      @      @        
�
predictions*�	   `/���    ͟�?     ί@!  `�w@�)�n(Gy"-@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.��>h�'��f�ʜ�7
������pz�w�7��})�l a��ߊ4F��f����>��(���>6�]��?����?�vV�R9?��ڋ?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?������?�iZ�?�P�1���?3?��|�?�������:�               @       @      @       @      *@      B@     �L@     @S@      W@     @Z@      \@      ]@      ]@     �\@      ]@     �`@      ^@     �V@     �Y@     @Y@     �T@     �M@     @T@      L@      G@      M@      N@     �I@      E@      C@     �B@      A@      =@      ;@      7@      8@      1@      6@      "@      ,@      "@      @      &@      1@      @      @      @       @      @      @      @      �?      �?      @      @      �?       @              �?      @       @      �?              �?       @      �?              �?      �?              @      �?       @              �?      �?              �?      �?              �?              �?              �?              �?       @              �?      �?              �?      �?              @      �?      �?      @       @       @       @      �?      @       @      @      @      @       @      @      "@      @      @      @      (@       @      .@      $@      &@      ,@      ,@      $@      ,@      1@      5@      :@     �B@      3@      ;@      ?@      7@     �@@      @@      E@      H@     �M@     �O@     �N@      H@      P@     �K@     @Q@      P@     �L@     �K@     �G@      D@     �C@      D@      ;@     �B@      H@      ;@      2@      <@      7@      1@      2@      1@      "@       @      "@      @      @      @      @      @      @      @      @      @              �?       @              �?              �?        
��F�'      ����	��u���A*�O

mean squared error�
,=

	r-squared�w�=
�7
states*�7	   @�0�   ��'@    ��=A!�Q/�x ��)��oIb�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W�H��'ϱS��
L�v�Q�28���FP���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J��#���j�Z�TA[���f׽r����tO����PæҭUݽH����ڽ�-���q�        ������=_�H�}��=�Bb�!�=�
6����=K?�\���=�J>2!K�R�>���<�)>�'v�V,>6NK��2>�so쩾4>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?      @      @      "@      *@     �A@     @Q@     �\@     �j@     p�@     ��@     ��@     ʭ@     ̴@     ��@    ��@    �L�@     ��@    ���@     +�@     �@    ���@    �$�@    �H�@    @��@    ��@    ���@    ��@    @��@    ���@    @��@     �@     �@     5�@     %�@    �5�@    �1�@    @��@    ���@     �@    @��@    �>�@    ���@    �d�@    @-�@     ��@     L�@     ��@     ��@    @i�@     C�@    ���@     ��@     D�@     w�@    ���@    @��@    ��@     	�@     "�@     d�@     S�@     ��@    ���@    �H�@     
�@    �A�@     M�@     b�@     �@     ��@     W�@     ��@     �@     �@     �@     ڪ@     ��@     ��@     J�@     $�@     ֠@     ��@     ��@     ę@     |�@     Д@     ��@     P�@      �@     ��@     �@     ��@     ��@     ��@     ��@     �@     `�@      |@     �{@      x@     �u@     �t@     0q@     @p@     �o@     �k@     �i@     �f@     @f@     �c@     �a@     �^@     �[@     �^@      ^@      Z@      R@     �R@     @T@     @P@      O@      N@     �Q@     �I@      F@     �A@      E@      9@      B@     �A@      ?@      @@      4@      ,@      5@      =@      ,@      1@      2@      .@      .@      $@      2@       @      @       @      ,@      "@      @      &@       @      @      "@      @      @      "@      @      @      @      @      @      @      @       @      �?      @      �?      @      �?       @      @      @      @       @      �?      @      �?       @       @      �?               @      �?      �?      �?               @               @      �?      �?      �?      �?              �?      �?              �?              �?              �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              @       @              �?              �?      �?              �?      �?              �?       @      �?      �?       @       @       @              �?      @      @       @      �?       @       @       @      @       @       @      @      �?      @      @      @      @      @      @       @      @      @      @      @       @      @      "@      @      @      @      @      @      $@      $@       @      @      .@      ,@      &@      *@      4@      ,@      @      1@      1@      .@      ?@      6@      ;@      8@      <@      C@      ;@      F@     �G@      E@      F@     �K@      M@     @P@     @R@     �M@      U@     �P@      Z@     �W@      [@     �Z@     �]@     @_@      ^@     �b@     �f@     �d@     �f@     �h@     @l@      p@     �p@     �r@      x@     @w@     @y@     y@      ~@     ��@     �@     ��@     Ѓ@     ؇@     ��@     ��@     ��@     ��@     l�@     �@     @�@     x�@     ��@     ��@     ,�@     �@     �@     �@     z�@     ��@     r�@     ,�@     ܯ@     ��@     s�@     �@     ��@     ��@     ��@     ��@     ��@     ��@    ���@     >�@     %�@    �!�@     ��@    �S�@    ���@    �^�@    �n�@    �~�@    @�@    �'�@     I�@    @��@    �y�@    �>�@    @��@    ���@     /�@    @��@    @p�@    ���@    @�@    �a�@    �Q�@     ��@    ���@    @�@     ��@     o�@    ���@    �X�@     ��@     �@    �q�@     I�@    �&�@     ^�@    ��@     ��@    ��@     x�@     ��@     .�@     ��@     �@     7�@     ��@    �>�@     ѻ@     �@     ;�@     (�@      �@     ��@     �@      i@      V@      B@      0@      &@      @       @       @      �?        
�
predictions*�	   ��g��   �	J�?     ί@!  ���5*@)6*I��3@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��.����ڋ��vV�R9��5�i}1���d�r�>�?�s���O�ʗ���I��P=��pz�w�7��K+�E��Ͼ['�?�;����?f�ʜ�7
?�5�i}1?�T7��?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�E̟���?yL�����?�������:�              @      "@      .@      7@     �D@     �F@     �W@     �[@     �Z@      _@      a@     @\@     �Y@     @Y@     �W@     @V@     �U@      Y@     �R@     �P@     �Q@      M@     �G@      K@      Q@     �B@     �F@      E@      @@      =@      ?@      5@      5@      8@      7@      .@      2@      *@      *@      0@      0@      (@      &@      @       @      @      @      @       @      @      @              @       @       @      �?      @       @      @       @      �?       @      @       @               @       @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      @               @               @       @      @      @      @      �?              @       @      @       @      @      @      @      @      @      @      "@      @      &@      (@      @      @      (@      1@       @      4@      *@      0@      0@      ;@      8@      5@      <@     �@@      @@     �A@      F@     �G@      E@     �G@      L@     �D@      G@      H@     �E@     �K@     �Q@     �@@      O@     �G@      P@      M@      B@      I@     �B@      G@      G@     �C@      >@      =@      8@      8@      5@      5@      0@      0@      .@      *@      @      "@      "@      @      $@      @      @      �?       @       @       @      �?              �?              �?              �?        �m�(      $��	c�Au���A*�O

mean squared error+�+=

	r-squared`�=
�8
states*�8	   �M��   �!�@    ��=A!�=�t�c��)Ǧ�`g�@2�w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP���T�L<��u��6
��K���7��[#=�؏�������~�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�:�AC)8g�cR�k�e������0c�w&���qa���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP��`�}6D�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,�4��evk'���o�kJ%�4�e|�Z#���-�z�!�Łt�=	���R����2!K�R������%���9�e����K����Qu�R"�PæҭUݽ�d7���Ƚ��؜�ƽG�L������6����@�桽�>�i�E�����_����e���]���1�ͥ��G-ֺ�І��-���q�        ��.4N�=;3����=�/�4��==��]���=��f��p>�i
�k>%���>��-�z�!>4��evk'>���<�)>7'_��+/>_"s�$1>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              @      *@      @       @     �@@     @Q@     �X@      m@     ��@     4�@     h�@     ��@     ��@     �@     z�@     �@     5�@     ��@    ���@    ���@     ��@    ���@    @
�@    �E�@    ���@    ���@    ���@     ;�@     D�@     ��@     ��@    �f�@     ��@    ���@    �I�@    ���@    @��@     �@    @�@    @'�@    �~�@    @��@    �g�@    �	�@     �@     %�@    ���@    @ �@    ���@    �w�@    @1�@     ��@    ���@     ��@    @�@    ���@     %�@    �7�@    �s�@    �`�@    ��@    ���@    ���@     ��@     ��@     <�@     q�@     �@     �@     7�@     �@     J�@     V�@     R�@     �@     ު@     �@     �@     >�@     v�@     Р@     ��@     ��@     ��@     \�@     4�@     ��@     (�@     ��@     ��@     0�@     X�@     ��@     P�@     ��@     (�@     h�@     �~@     �{@     px@      x@      t@     �r@     `r@     �p@     �m@     �i@      k@      f@     `d@     @a@      `@     @a@     �^@     @^@     @[@     �Y@      T@     �T@     �J@      R@     �V@     �J@      I@     �H@      G@     �C@     �@@      A@     �B@      C@      9@      =@      @@      ;@      8@      ?@      5@      2@      @      3@      ,@      $@      *@      $@      @      @      "@      $@       @      @      &@      @      @      @       @      "@       @       @      @      @      @      �?      @      �?      @      @              @      @      @              �?      @       @      @               @       @      �?      �?              �?              @      �?      �?              @       @      �?              �?      �?              �?      �?               @      �?              �?              �?               @              �?              �?      �?               @      �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?      �?      �?              �?              �?              �?      �?               @       @      �?      �?      �?      �?               @              @      �?      �?      �?      �?       @      �?       @      �?      �?      �?      �?      @              @       @      @       @       @      @      @      �?              �?      @      @       @      @      @       @      @      @      �?      @      @      @      @       @      @      $@      ,@      &@      ,@      (@      0@      "@      (@      .@      1@      .@      1@      5@      4@     �@@      :@     �@@      >@      >@     �B@      <@      E@      G@     �H@     �H@      I@      N@      N@     �R@      S@     �R@     �S@      Z@     �T@     �W@     @Y@      `@     @a@      `@     �c@     �e@     �c@      g@      i@     �n@     �p@     �r@     Pt@     �t@     0u@     `y@      y@     �|@      @     Ѐ@     ��@     ��@      �@     �@     ��@     ��@     l�@     ��@     ��@     ��@     �@     �@     p�@     `�@     ��@     4�@     |�@     ʥ@     Ψ@     ��@     @�@     +�@     ݱ@     ѳ@     ��@     �@     �@     ��@     \�@     B�@     ^�@     e�@    ���@    ���@     �@     o�@    ��@    @a�@    ���@    ���@     ��@     ��@    ���@     ��@     2�@    �Q�@    �(�@    �m�@    ���@    �W�@    ��@    @j�@     ��@    ��@    �o�@     C�@    @��@    �i�@    ���@     y�@     ��@     k�@     ��@     ��@    �2�@     f�@     ��@    ��@     !�@     ��@    ���@    ���@     -�@    ���@    ��@     ��@    �_�@    ���@    ��@     ��@     $�@     ��@     6�@     ��@     4�@     ޡ@     �@     �i@     �V@      D@      4@      3@      @      @       @       @        
�
predictions*�	   �-�ſ   �C1�?     ί@!  #f��B�)W�b)Y*@2��QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����d�r�x?�x��0�6�/n���u`P+d���_�T�l�>�iD*L��>I��P=�>��Zr[v�>x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?��Z%��?�1%�?\l�9�?+Se*8�?�E̟���?yL�����?�������:�              �?      �?      �?      �?      @      @      3@      6@      >@      H@     �R@     @Z@     ``@     �b@     `f@     @d@     @b@     �`@     `c@     �`@      c@     �_@      `@     �X@      V@     �S@     �M@     �H@      J@     �B@     �K@     �I@     �A@     �C@      8@      :@      2@      9@      0@      3@      ,@      2@      2@      ,@      "@      .@      @      "@      @      $@       @      @      @      @      @      @      @      @       @      @      @       @       @      @              �?      �?              �?       @      �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @      �?      �?      �?      �?       @       @      @      �?       @      @      @      @      @       @      @      @       @      @      $@      @      @      &@      @      "@      (@      @      ,@      $@      (@      1@      ,@      3@      2@      0@      9@      4@      6@      @@      2@     �@@      ;@      D@      ?@      3@      =@      ;@      B@      3@     �F@      C@      A@      >@      B@     �E@     �D@     �A@      C@      ;@      <@      :@      2@      2@      3@      &@      (@      1@      (@       @      ,@      "@       @       @      @      @      @      @       @               @      �?      �?              �?        e�]�)      |/��	Lnu���A*�S

mean squared error�(=

	r-squaredT{>
�;
states*�;	    ~W�   @��@    ��=A!l�/�����)�ܩ�)�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���#���j�Z�TA[��nx6�X� ��f׽r����tO����f;H�\Q������%��'j��p���1����Qu�R"�PæҭUݽ��
"
ֽ�|86	ԽG�L������6������/���EDPq��y�訥�V���Ұ���-���q�        !���)_�=����z5�=y�訥=��M�eӧ=(�+y�6�=�|86	�=PæҭU�=�Qu�R"�=f;H�\Q�=�tO���=�f׽r��=�`��>�mm7&c>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?      @      @      "@      $@      .@      E@      Q@     `e@     p|@     t�@     Ц@     ��@     ٳ@     `�@     �@    ���@    �&�@    �^�@    ���@    ���@     ��@     ��@     4�@    �B�@    ���@    @_�@    �Z�@    ��@     ��@     �@    ��@     ��@    ���@     �@     ��@    �W�@    ���@    @K�@    @��@     3�@    ���@    @$�@    �I�@    ���@    �l�@     ��@    �0�@     ��@     ��@     :�@     ��@    �3�@    ���@    ��@    ���@    �m�@     c�@    ���@     ��@    ���@    �.�@    ���@     ^�@     �@    ��@    ��@    �d�@     |�@     '�@     ��@     s�@     ��@     3�@     ��@     r�@     <�@     �@     �@     ��@     \�@     l�@     ^�@     x�@     p�@     ��@     ȗ@     T�@     �@     <�@     ��@      �@     ��@     ��@     H�@     0�@     ��@     ��@     (�@     �|@     �x@     �x@     w@     �u@     �t@     �q@     pp@      n@     �e@     @h@     �e@     �e@     `a@      d@     ``@     �Z@     �[@     �U@     �T@     �Y@     @T@     @R@     �Q@      N@     �I@     �I@     �M@      L@     �A@      >@     �C@      @@      C@      >@      A@      8@      5@      *@      :@      7@      6@      8@      ,@      4@      ,@      &@      (@      @      0@      ,@      (@      $@      @      @      @      @       @      @      @      @      @      @       @       @      @      @              @      @       @      @              �?      �?      �?       @      @      �?              �?      �?       @      �?      @      @               @              �?      @       @              �?      �?      �?              �?      �?               @               @       @      �?              �?              �?      �?              �?      �?              �?      �?              �?      �?              �?              �?      �?      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?       @      �?      �?               @              �?      �?      �?      @              �?               @      �?              �?      �?              �?              �?       @      �?       @      �?      @      �?      @      �?      �?       @      @      �?              @      @      �?              @      @       @      @       @      @      @      @      @      $@      @       @      (@       @      @       @      @      $@      @      $@      @      &@       @      ,@      0@      7@      "@      9@      5@      7@      3@      2@      7@      6@      7@     �@@      @@      @@      E@      @@      C@     �F@     �I@     �M@     �F@     �K@      O@      Q@     �H@     �Q@     @V@     �W@     �Z@     @[@     �[@     �`@     @a@     @b@      h@     @d@     �e@      h@     �m@     p@     �m@     �r@     Ps@     �s@     �v@     Pw@     �|@      ~@     p|@     �@     `�@     ��@     �@     X�@      �@     h�@     4�@     Б@     ��@     �@      �@     p�@     �@     �@     �@     d�@     Τ@     �@     T�@     ��@     $�@     \�@     ��@     D�@     ��@     ��@     -�@     ��@     ս@     ��@     )�@    �t�@    �G�@     ��@     ��@     {�@    ���@    �d�@    �B�@    �\�@     N�@     �@    @��@     ��@    @$�@    @-�@    @R�@     ��@    �+�@    �1�@    ���@     }�@     ��@    @-�@    @��@    @J�@    ���@     ��@    �e�@    �8�@    ���@    �n�@    ���@     C�@     ��@    ���@     Y�@    �8�@    ���@    �)�@     ��@     ��@    ���@     ��@    ���@    ���@    ���@     ��@     Y�@    �w�@    ���@     b�@     j�@     �@     H�@     ��@     ��@     �@     �k@     @S@      B@      (@      @      @       @              �?        
�
predictions*�	   @�ÿ    Km@     ί@!  g�SL@)[D���5@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9��x?�x��>h�'��6�]���1��a˲���[���FF�G ������>
�/eq
�>�f����>��(���>��[�?1��a˲?6�]��?��d�r?�5�i}1?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?+�;$�?cI���?��tM@w`<f@�������:�               @      �?      @      @       @      &@      (@     �A@      ;@     �E@      J@      L@      T@     @Q@      T@     �X@     @U@     @U@      S@      S@     �P@      T@      Q@     �I@     �M@      J@     �K@     �E@      E@      B@     �F@     �B@      2@      5@      *@      ;@      1@      0@      ,@      (@      &@      "@      @      "@      @      @      (@      $@      @      @      $@      "@      @      @      @      @      �?      @      @      @      @      �?              @      �?       @      �?      @      �?               @              @              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?       @       @       @       @              �?      �?      @       @       @      �?      @      @      @      @      @      @       @      "@      "@      "@      @       @       @      *@      @      .@      (@      1@      *@      2@      3@      4@      9@      ;@      8@      @@      7@      C@     �C@     �H@      G@      P@     �P@     �M@     �N@     �P@     �U@     �U@      R@     �T@     @\@     @T@     @U@     @R@     �T@      R@      O@      O@     �D@     �D@      L@      B@      =@     �F@     �A@      ;@      *@      $@      3@      *@      &@      &@      "@      @      @      @      @      @      �?      �?       @      �?      @              �?              �?              �?        �-X)      "�F�	�*�u���A*�Q

mean squared error��%=

	r-squaredt>
�9
states*�9	   ����    ��@    ��=A!;c0 ���)�Quu�)�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,�4�e|�Z#���-�z�!�%�����i
�k��J��#���j�Z�TA[�����"��`���nx6�X� ��f׽r����tO������-��J�'j��p�G�L������6��|_�@V5����M�eӧ�����z5��!���)_���-���q�        ���6�=G�L��=5%���=�Bb�!�=i@4[��=z�����=ݟ��uy�=�/�4��=Łt�=	>��f��p>�i
�k>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>�so쩾4>�z��6>u 5�9>/�p`B>�`�}6D>��Ő�;F>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?      @      "@      &@      *@      <@      G@     @V@     ps@     ��@     ��@     Ψ@     ^�@     ʴ@     ��@     ս@    ���@    �M�@    ���@    ���@     ��@     ��@    ���@     ��@     ��@     e�@     ��@     ��@     ��@    ��@     ��@    ���@     ��@    �Y�@    @X�@     ��@    �#�@     V�@    ��@    �7�@    ���@    ���@    @��@    @,�@    ���@    �t�@    �2�@     $�@     ��@    ���@    @��@     ��@    @b�@    ���@    ���@    ���@     �@    �8�@    �e�@    ���@     8�@     ��@    �I�@    ���@     ��@    ���@     ��@     ��@     ׻@     ��@     �@     ִ@     ��@     >�@     ��@     ��@     v�@     ��@     x�@     ��@     ��@     Z�@     �@     ��@     ��@     ��@     |�@     ��@     ��@     ��@     0�@     ��@     ��@     ��@     (�@     p�@     (�@     Ѐ@     P~@     0z@     �y@     �v@     �t@     ps@      t@      r@     �o@     �f@     �g@      f@     �e@     �b@     �e@     �a@     �`@     �_@     �\@     �Y@      X@      T@     @Y@     �T@      P@     �H@     �J@     �G@      L@     �F@     �G@      =@      @@      @@      <@      ?@      5@      ;@      5@      9@      0@      2@      ,@      1@      2@      9@      &@      "@      (@      ,@      $@      (@      &@       @       @      $@       @      $@      @       @      @      @      @      @      @      @      @       @      �?      �?       @       @      @       @      @      �?      �?       @               @      @      @      �?      �?      �?              �?      @      @               @              @       @       @      �?              @              �?              �?              �?      �?      �?              �?      �?               @              �?              �?              �?              �?               @               @              �?              �?              �?              �?               @              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              �?      �?              �?      �?               @       @      �?              �?              �?      �?      �?      �?      @       @              @      �?      @       @      @      �?       @      @      �?      @      @       @       @      @      @      @      @      @      @       @      @      @      @      @      @      "@      @      @      @      $@      "@      "@      "@      @      @      0@      "@      .@      (@      4@      .@      6@      .@      0@      2@      3@      7@      8@      5@      6@      4@      9@     �A@     �A@      >@      @@     �@@     �I@     �F@      I@     �H@     @R@      M@     �L@     @P@     �T@     �U@     �Z@     @Y@     @Y@     �X@     �`@     @]@     �c@     �b@     @i@     `i@     �i@      i@      o@     �m@     pr@     �t@     0u@     �v@     @z@     �}@      ~@     �@     ��@     p�@     8�@     ��@     ��@     X�@     �@     `�@     ��@     0�@     H�@     ��@     �@     <�@     �@     �@     ,�@     ��@     (�@     Ω@     L�@     ��@     H�@     �@     �@     ��@     ��@     N�@     ��@     S�@     ��@     ��@     ,�@     /�@    ���@     ��@     ?�@    ���@    ��@    ���@    �I�@    ���@    @e�@     E�@    ���@    ���@    �c�@    ���@     ��@    ��@    �h�@    ��@    ���@     ��@    @��@     ��@    @��@     ��@     ��@    �B�@    @��@    @J�@    ���@    �`�@     I�@    ���@    ���@    ���@     /�@    �C�@    ���@    �9�@    ���@     G�@     )�@    ���@    �}�@     ��@    �%�@    �K�@     3�@     K�@     �@     r�@     `�@     
�@     �@     �@     �g@      Q@      G@      6@      $@       @       @      �?      �?        
�
predictions*�	    ��ǿ    c�?     ί@!  ���7�)��j멻9@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���[���FF�G �['�?��>K+�E���>�ߊ4F��>})�l a�>>�?�s��>�FF�G ?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?�������:�              �?              �?      @      &@      2@      4@      ;@     �G@     �Q@      X@     @Z@     @b@     @c@      e@      b@     @b@      _@     �\@      W@     �U@     �S@      T@      W@     @V@     �Q@      I@     �K@     �C@      K@     �I@      @@      C@      @@      ?@      ;@      :@      2@      ;@      9@      ;@      &@      6@      (@      $@      &@      @      $@      @      (@       @       @      "@      @      @      @      @      @      @      �?      �?       @      �?      �?       @      �?               @      @      �?               @       @      �?              �?              �?              �?              �?               @              �?      �?               @              �?      �?      �?      �?      �?      �?       @      �?      �?      @              �?      @      �?      @      �?       @      @       @       @      @      @      "@      "@      @      @      @      "@      "@       @      $@      &@      &@      @      ,@      4@      ,@      "@      3@      .@      ;@      >@      E@      0@      D@     �@@     �E@      ?@      C@      F@      G@      G@     �A@     �@@     �M@      ?@      B@     �@@      M@     �C@      B@      D@      8@     �@@      B@      5@      @@      0@      0@      1@      0@      .@      $@      $@      *@      @      "@      @      @      @      @       @      �?      �?      �?      �?       @       @      �?       @      �?              �?      �?              �?        �{G6r*      �G�3	�g�u���A*�T

mean squared error��&=

	r-squared��>
�<
states*�<	   @iU�    B� @    ��=A!�L��)��).0� ��@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6����<�)�4��evk'�4�e|�Z#���-�z�!�%����2!K�R���J��#���j�Z�TA[��RT��+��y�+pm�nx6�X� ��f׽r����tO�����K��󽉊-��J���1���=��]���;3���н��.4Nν�EDPq���8�4L���V���Ұ����@�桽�>�i�E�����:������z5����x�����1�ͥ��̴�L�����/k��ڂ�        �-���q=e���]�=���_���=��
"
�=���X>�=��1���='j��p�=��-��J�=�K���=�9�e��=�f׽r��=nx6�X� >�`��>RT��+�>���">Z�TA[�>�#���j>��R���>Łt�=	>��f��p>%���>��-�z�!>��o�kJ%>4��evk'>�'v�V,>7'_��+/>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@�������:�              �?      @      @      (@      (@      :@      N@      _@     �z@     �@     V�@     d�@     2�@     �@     �@     ��@    ���@     �@     �@     ��@     ��@     ��@     ��@    �O�@    ���@     W�@    ���@    ���@    �"�@     ��@     �@    ���@     =�@    ���@     ��@     B�@     ��@    ���@    @f�@    @J�@    ���@    @+�@     F�@    ���@    ���@     ��@    �M�@    ���@    @o�@    ���@    �]�@    ���@    @(�@     ��@     w�@    ���@    ���@    ���@     :�@     �@    �N�@     .�@     ��@    ���@    ���@     ��@    ��@    ���@     (�@     ��@     ��@     �@     �@     }�@     �@     ��@     ��@     P�@     H�@     �@     ��@     ��@     �@     *�@     ��@     ��@     ��@     ��@      �@     (�@     L�@     H�@      �@     ��@     ��@     x�@     x�@     ��@     0�@     �@     @     �}@     �y@     �v@     �v@     �s@     �p@     0s@     �p@     �k@     �k@     `f@      d@     �c@     �b@     �a@      ^@      [@     �[@     �^@      U@      Y@     �S@     @R@      P@     �R@     �L@      O@     �F@     �F@      C@      ;@     �@@      B@      >@      0@      =@      :@      >@      2@      1@      7@      3@      1@      8@      ,@      &@      &@       @      :@      1@      @      @      "@      (@       @      @      @      @       @       @      @      &@      "@      @      @       @      @      @      @      @      @      �?      @      @      @       @              �?      �?      @      �?      @               @      �?      @      �?      �?               @       @      �?       @              @              �?       @      �?       @       @       @      �?      �?              �?      �?               @              �?               @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?      �?               @      �?              �?              �?              �?              �?      �?              �?              �?      @              �?      �?      @              �?       @       @       @              @      �?       @      @      �?      @      @      @      @      @      �?       @      @      @      �?      @      �?      �?      @      @      �?      @      @      @      @      @      @      @      "@      @       @      "@      @      @      @      @      @       @       @      $@      $@      &@      *@      (@      .@      2@      *@      4@      0@      1@      6@      5@      .@      5@      6@      4@      ;@      :@      9@     �A@      ;@      ?@     �F@     �E@      I@      J@      F@      I@     �L@     @P@     �R@     �R@     �R@      S@      Z@      R@     �^@     �`@      `@     @a@     @e@     �f@     �c@     �f@     �k@     @j@     �o@     0q@     �q@     �t@     �v@     pw@     �{@     @|@     �~@     ��@      �@     @�@     X�@     H�@     ��@     P�@     x�@     ��@     ��@     ��@     P�@     L�@     �@     Н@     ��@     �@     �@     �@     ��@     ��@     ��@     �@     Z�@     Ʋ@     U�@     Ѷ@     ��@     ��@     e�@    �,�@     ��@    ��@    ���@    �?�@     Z�@     :�@    �R�@     ��@    @ �@     �@     �@     �@    @��@     ��@    �>�@     Y�@    ���@     ��@     l�@    ���@    @�@    ���@    @�@    ���@    ���@     �@     ��@    @q�@    ���@    ���@    @ �@    �
�@     ��@    �F�@    ���@    � �@     @�@     ��@     �@     >�@    ���@     �@    ���@     ��@    �]�@     �@    �"�@     ��@    �l�@    �)�@    �D�@     Ͻ@     z�@     �@     h�@     ��@     ��@     �@     `|@      d@     �U@     �A@      5@      .@      @      �?        
�
predictions*�	   ��ƿ   �>\@     ί@!  pFCy/@)	���@3@2��QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9�O�ʗ�����Zr[v��I��P=��pz�w�7���ߊ4F��h���`[�=�k���*��ڽ�6�]��?����?>h�'�?x?�x�?ji6�9�?�S�F !?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?w`<f@�6v��@�������:�               @      @      @      @      @      &@      4@      5@      @@      D@      K@     @Q@     �S@     �Q@     �Z@     @T@     �X@     @T@     @U@      U@     �S@     �T@     @R@     @T@     �P@      R@      K@     �P@      N@      K@      :@      >@     �A@      >@      6@      =@      2@      1@      ,@      3@      8@      5@      (@      (@      $@      *@      "@       @      $@      $@      @      @      @      @      @      @              @      @      �?       @      @      @      @      @      @      �?      �?      @      �?              �?      �?      �?      �?      @      �?               @              �?              �?              �?              �?              �?              �?              �?              @      �?      @      �?              �?       @      @       @      @      @       @      @              �?       @      @      @      @      @      @      @      (@      @      "@      $@       @      .@      (@      &@      @      3@      1@      1@      9@      7@      8@      6@      7@     �@@      @@      B@      J@      J@     �E@     �H@     �K@      N@      R@     @P@     �P@     @R@     �S@     �V@     @R@     �R@     @R@     �O@     �Q@      L@      I@     �J@      B@      ?@     �A@      ?@      <@      ,@      $@      4@      (@      @      @      &@      @      @      @       @      �?      @      @       @      @       @               @       @              �?              �?        ?�b+      �h)	���u���A*�U

mean squared errorZ�"=

	r-squaredTm.>
�<
states*�<	    �|�   @��@    ��=A!�iHKM��)wTA�4�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!���R����2!K�R���J��#���j�y�+pm��mm7&c��`���nx6�X� ��tO����f;H�\Q���9�e����K���(�+y�6ҽ;3���н�d7���Ƚ��؜�ƽ�1�ͥ��G-ֺ�І�̴�L�����-���q�        K?�\���=�b1��=�d7����=�!p/�^�=H�����=PæҭU�==��]���=��1���=�9�e��=����%�=f;H�\Q�=nx6�X� >�`��>�mm7&c>���">Z�TA[�>�#���j>��R���>Łt�=	>��f��p>�i
�k>%���>4�e|�Z#>��o�kJ%>_"s�$1>6NK��2>�so쩾4>�z��6>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�               @      �?      $@      @      *@      H@      O@     �i@     Є@     8�@     ħ@     H�@     ��@     ��@     �@     ��@     ��@     ��@     ��@     ��@    �v�@     ��@    ���@     O�@     �@     ��@    ���@    �X�@    �x�@    �e�@     n�@     �@    ���@      �@     z�@     �@    �b�@    ���@    ���@    �7�@    ���@    ���@    �T�@    ��@    ��@    ���@    @"�@    �w�@    �e�@    @*�@     ��@     ��@    @��@    @��@     ��@    �*�@    ���@     ^�@    ���@    ���@     ��@     ��@     /�@     M�@    �G�@    �D�@    ���@    �y�@     �@     %�@     ǹ@     ��@     ۵@     N�@     �@     o�@     �@     ��@     ��@     .�@     ��@     @�@     ��@     �@     t�@     �@     �@     ��@     �@     $�@     Ȑ@     ��@     H�@     ��@     (�@     p�@     ��@     �@     �@     0@     �{@      y@      x@     Pv@     �t@     �r@      q@     �n@     �l@     �o@     `h@      f@      g@     �a@     �`@     �`@     @^@     �X@     �T@     �U@     �W@     �T@     �P@      O@      P@     �I@     �J@     �E@     �H@      F@     �B@     �F@      <@      A@      ?@      @@     �@@      :@      :@      5@      5@      *@      (@      7@      1@      0@      *@      @      (@      *@      *@      ,@      "@      @      .@       @       @      @      @       @      @      @      @      @       @       @      @       @      @      @      @      @      �?      @      �?      @       @       @      @      @      �?      @              �?       @               @      �?       @      @      @      @      �?              �?      �?       @      �?      @      @       @              @              �?      @      �?              �?      �?      �?              @              �?       @              �?      �?              �?               @              �?              �?               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?       @              �?              �?              �?       @       @              �?              �?      @              �?      �?      �?               @               @      �?               @              �?      �?      @      �?               @       @      @      @       @      �?       @      �?       @      @       @              �?              @       @      @       @      @      @      @      @      $@       @      @      "@       @      @      "@      @       @      "@       @      "@      "@      (@      *@      .@      (@      1@      0@      *@      0@      5@      7@      4@      2@      :@      4@      7@      :@      A@      6@     �A@      D@     �A@     �E@      E@      N@     �H@     �G@     �R@     �P@      R@      Q@     @S@     �W@     �U@     @X@     �`@      [@      [@      c@     �b@      d@     `h@      i@      j@     �m@     @n@     s@     `s@     �s@     u@     �z@     �z@     @@     0�@     �@      �@     Є@     ��@      �@     p�@     H�@     $�@     ��@     ��@     ��@     �@     ș@     �@     �@     ~�@     �@     ��@     6�@     \�@     ��@     `�@     ��@     ��@     �@     �@     �@     ��@     ��@     k�@    ���@     V�@    ��@    ���@    ���@    ��@     ��@     h�@    �O�@    �/�@    �w�@    @��@    @��@     7�@    @��@    �A�@    �*�@    @T�@    @*�@    ���@     ��@    @��@     ��@    @p�@     ��@    ���@    @a�@     �@     ��@    �T�@     ��@    @
�@     ��@    �j�@    ���@    ���@    ���@     Y�@    �4�@     O�@    ���@    ���@    �L�@    �}�@    ���@     @�@    ���@     6�@    �.�@    ��@     Ŀ@     ü@     ��@     m�@     ��@     6�@     ��@     ~�@     �~@      g@     �T@      D@      3@      0@      @      @      �?        
�
predictions*�	   @��Ͽ   `q{@     ί@!  �֕}�)��R#�5@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7����5�i}1���d�r�>h�'��f�ʜ�7
������6�]�����(��澢f���侢f����>��(���>��Zr[v�>O�ʗ��>��[�?1��a˲?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?+�;$�?cI���?�P�1���?��tM@w`<f@�������:�              �?              �?      �?       @      @      @      *@      1@      *@      $@      <@     �E@     �N@     @R@     @S@     @U@     �`@     �b@     @]@     @^@     �^@     �_@      \@     �\@     �W@     @U@     �W@     �Q@      K@     �G@      G@     �J@     �G@      <@      @@      7@     �E@      8@      1@      3@      7@      &@      .@      ,@      0@      "@      5@      "@       @       @      &@      &@      @      @      �?      @      @      @      @      @      �?      �?      �?      �?      @       @       @      �?      @       @       @              �?      �?      �?              @       @              �?              �?               @              �?              �?              �?              �?              �?              �?              �?      @      �?              @       @               @      �?      �?      @      �?      @               @              �?      @      �?       @      �?       @       @      @      @       @       @       @      @      @      @      @      @      .@      &@      (@      $@      (@      .@      &@      7@      2@      4@      ;@      2@      :@      <@      7@     �A@     �@@      =@     �F@     �E@      F@     �G@     �F@     �M@     �L@     �J@     �A@      G@      N@     �H@      J@     �C@      G@      K@     �F@     �G@     �C@     �C@      ?@      @@      <@      6@      ,@      2@      5@      *@      *@      ,@      &@      @      @      @      @      @      @       @      �?      �?       @       @      �?              �?      �?              �?        �H)�")      "�x�	 �v���A*�R

mean squared errorfD =

	r-squared�6;>
�:
states*�:	   @V_�   ��@    ��=A!���l%<��)Ow@�J�@2��6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�2!K�R���J��#���j����"�RT��+��nx6�X� ��f׽r����tO����f;H�\Q������%���K��󽉊-��J�i@4[���Qu�R"����X>ؽ��
"
ֽ����/���EDPq��\��$��%�f*��-���q�        5%���=�Bb�!�=ݟ��uy�=�/�4��=��1���='j��p�=��-��J�=����%�=f;H�\Q�=�`��>�mm7&c>���">Z�TA[�>�#���j>6NK��2>�so쩾4>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�������:�              �?      @      $@      (@      =@      ?@     �S@     `l@     X�@     ԕ@     ��@     ڬ@     Ǳ@     `�@     �@     $�@    �S�@     ��@     ��@    �
�@     ��@    ���@     �@     �@    ��@    �Z�@    �R�@     ��@     �@    ���@    ��@     |�@    �k�@     ��@    �>�@     j�@    ���@     ��@    �/�@    ���@    @a�@     Y�@    @��@    ���@     ��@    �'�@     f�@    @��@    ���@    ���@    �V�@    @��@    �~�@    @��@     K�@     \�@    ���@     T�@    �U�@     ��@    ���@    �|�@     ��@    ���@     ��@    �L�@     ��@    �#�@     Ž@     ޻@     �@     Ķ@     ��@     I�@     W�@     M�@     ��@     ��@     �@     ޥ@     ��@     `�@     n�@     t�@     T�@     ��@     ��@     0�@     ��@     ��@     �@     ȏ@     (�@     �@     ��@     X�@     0�@     ��@     �@     �~@     p{@     Pz@     �x@     @t@     �s@     @s@     @m@     �n@      l@     @k@      i@     �d@     �g@     �b@     �a@      _@     �[@     �\@     �Z@     @W@     @S@      U@     @S@     @S@      M@     �N@      N@      I@     �F@     �I@      D@     �D@     �B@     �B@      @@      @@      9@      3@      :@      3@      8@      *@      3@      1@      &@      $@      7@      1@      "@      *@      1@      @      @      "@      @      (@      "@      @      @      @      �?      @       @      @      @      @      @      @      @      @       @      �?      @      @              @      @      @      @      @      @      @      @       @      @               @      �?       @              �?      �?              @      �?      @       @       @      �?       @       @      �?              �?              �?      �?      �?      �?               @       @              �?      �?               @              �?               @       @              �?              �?      �?              �?              �?               @              �?              �?              �?               @              �?              �?              �?      �?              �?               @              �?      �?              �?              �?      �?              �?               @      �?      �?      �?      �?              �?      �?      �?       @              @       @      @      �?      @       @       @      �?              @       @      @      @       @      �?      �?       @      @      @      �?      @      @      @       @      @      @      @       @       @      @      @      @      @      $@      (@      2@      &@      &@       @      @      *@      "@      ,@      *@       @      ,@      1@      .@      2@      6@      4@      8@      0@      5@      2@      @@      :@      2@     �@@      7@      @@     �B@      I@     �K@      G@      I@      P@     �J@     @P@     �R@     �Q@     �T@     @U@     @W@      \@     �X@      [@     �^@     �a@      c@     �f@     @g@     �g@     �i@      l@     �q@     0s@      q@     @t@     �x@     �w@     Py@     �}@     �@     ��@     p�@     ��@     @�@     ��@     @�@     ȍ@     h�@     H�@     ��@     ��@     �@     �@     l�@     �@     j�@     Т@     �@     Ħ@     �@     �@     n�@     i�@     �@     ��@     ٵ@     Է@     6�@     _�@     4�@    ��@     ��@     a�@    �V�@     p�@    ���@    ���@    ���@     ~�@     ��@     ��@    @��@    ���@    �`�@    ���@     ��@    �)�@    @�@     .�@    @��@     ��@    �Q�@    ���@    @Q�@    @ �@    @|�@     v�@     ��@    ���@    @��@    �G�@    @��@     x�@    ���@    ���@    �b�@     /�@    ��@    ���@    ���@    ���@     q�@    ���@    �e�@    �r�@    �j�@     ��@     E�@     ��@    �n�@     G�@     L�@     ��@     ֵ@     ��@     $�@     ��@      �@     ��@     `h@     @X@     �I@      =@      ,@      (@      @      �?        
�
predictions*�	   ���˿    [	@     ί@!  �/U,@)HQ��[�@@2��K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82��[^:��"��S�F !��.����ڋ��[�=�k�>��~���>��[�?1��a˲?x?�x�?��d�r?�T7��?�vV�R9?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?cI���?�P�1���?�E̟���?yL�����?h�5�@�Š)U	@�������:�              �?       @              @      @      @      &@      (@      7@      5@      A@      M@      R@      V@     �Z@     @\@     ``@      `@     @_@     �a@     �a@     �\@     @Z@     @W@     �W@     @R@     �M@     �Q@      J@     �E@      L@     �K@      D@     �@@      :@      ?@      8@      3@      3@      6@      :@      1@      2@      "@      *@      &@      ,@      $@      @      $@      "@      @      @      �?      @       @      @      @       @      @      �?       @       @       @      �?      @      �?      �?       @      @              �?              �?              �?              �?              �?              �?              �?      �?               @      �?      �?      �?      �?      �?      @      �?      �?              �?      �?       @      @      @      @       @      @      @      @      @      @      .@      $@      @      ,@       @      .@      .@      *@      7@      9@      ,@      ,@      6@      5@      2@      >@      <@      ?@      3@      D@      L@     �J@     �C@      H@     �I@      =@     �D@     �F@     �D@      ?@     �G@      D@      D@      F@     �H@      I@     �E@     �C@     �C@      A@      =@      >@      7@      ;@      7@     �A@      7@      ,@      $@      9@      *@      $@      &@      @      @      @      @      @      @      �?       @      �?       @              �?              �?              �?        �e��,      :��	A*Ev���A*�Y

mean squared error�,=

	r-squared��U>
�@
states*�@	    +��   ��6@    ��=A!���L����)�M���j�@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p����"�RT��+���mm7&c��`���nx6�X� ��tO����f;H�\Q���9�e����K���'j��p���1���=��]����/�4��ݟ��uy�z�����PæҭUݽH����ڽ�|86	Խ(�+y�6ҽ;3���н��.4Nν��s�����:������z5�����_����e���]��̴�L�����/k��ڂ�x�_��y�'1˅Jjw��-���q�        ��@��=V���Ұ�=�EDPq�=����/�=�Į#��=���6�=��.4N�=;3����=(�+y�6�=�|86	�=i@4[��=z�����=ݟ��uy�='j��p�=��-��J�=�K���=�9�e��=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              �?       @      $@       @      0@      :@     �C@     �W@     �n@     Ѕ@     H�@     h�@     4�@     R�@     �@     8�@     �@    �s�@     ��@    ��@    ��@     ��@    ���@     n�@     m�@     g�@     ��@    �T�@    ���@     �@    ���@    �@�@     �@     ��@     ��@     ~�@    ���@    ��@    ���@     ��@     7�@    ���@    @��@    ��@     ��@     b�@    @A�@    ���@     g�@    ���@    �-�@    ���@    ��@    @�@    ���@    @��@    ���@     ��@    ���@    ��@     �@     p�@     ��@     ��@     �@     �@    �R�@    ���@    ��@     ��@     c�@     _�@     �@     �@     V�@     ��@     ��@     �@     h�@     0�@     ��@     r�@     �@     &�@     ��@     ��@     �@     ��@     �@     L�@     L�@     ��@     ��@     ��@     ��@     0�@     H�@     ��@     ��@     ��@     0~@     0|@     �z@     �x@     �u@     �t@      s@      o@     `p@      o@      j@     �i@      g@     `g@     �b@      `@      a@     @`@     @\@     @]@      V@     �W@     �S@     �T@     @Q@     @P@     @P@      G@     �J@     �I@      ?@     �I@     �H@      :@     �B@      E@      :@      >@      9@      9@      5@      7@      4@      :@      *@      4@      4@      0@      *@      (@      "@      0@      *@      @      @      &@      @      @      &@      @      "@      @      @      @      @       @      $@      @      �?      @      @      @      @       @      @      @      @      @       @      @      @      @      �?       @              @      @      @       @      @       @      �?      @      �?       @      @              @      �?       @      �?       @       @      �?      �?              @       @       @               @              �?      �?       @               @      �?      �?       @      �?               @              �?              �?       @              �?              �?              �?      �?              �?      �?              �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      �?      �?       @               @              �?      �?      �?      �?              �?              �?              �?              �?              �?      �?      @              �?      �?       @              �?      �?              �?      @      @       @              �?      �?               @       @      @               @      �?      @      @      �?       @      @      @      @      @      @      @              @      @      @      @      "@      @       @      @       @      @      @      @      @      $@      @      @      (@      @      *@      $@      (@      ,@      @      .@      *@      ,@      *@      *@      (@      7@      8@      3@      4@      8@      ;@      6@      <@      ?@      >@     �B@      D@      F@     �F@     �E@      ?@     �L@     �E@      I@     �K@      P@      L@     @Q@     �R@     �R@      Y@     �X@     @Y@      ]@     @]@     �_@      b@     �f@     `e@     `h@     �h@     �m@     @n@     �o@     �p@     Pu@     �w@     �x@     �x@      |@      ~@     0�@     �@     @�@     x�@     X�@      �@     �@     �@     0�@     ��@      �@     X�@     ��@     T�@     ��@     ��@     V�@     ��@     ��@     x�@     ~�@     ��@     ��@     �@     /�@     ��@     8�@     4�@     �@     M�@     ��@    ���@    ���@    �Q�@    ��@     ��@    ���@    �a�@    ���@    @4�@     ��@     ��@    @��@    ���@    ���@    @��@    @��@     1�@    @ �@    ��@     ��@    �~�@    ���@     ��@    @��@     \�@    @�@    �|�@    ���@    @��@    �t�@    ��@    ���@    @�@    ��@    �$�@    ���@    �&�@    ��@    �'�@    ���@     ��@    ���@    �e�@    ���@    �)�@    �`�@     r�@     ��@     �@    ���@     ��@     Ի@     �@     �@     �@     R�@     �@     ��@     0�@      n@     @Y@     �M@      3@      4@      ,@      @      @      �?      �?        
�
predictions*�	    �SϿ   @,J@     ί@!  ��E�)�ٖ��=@2����ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��������Zr[v��I��P=��1��a˲?6�]��?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?h�5�@�Š)U	@�������:�              �?              �?      @       @      @      @      &@      (@      3@      8@     �D@     �E@      Q@      S@     �S@     �T@     �X@     @Z@     �^@     @Z@     �`@     @]@     �Y@     @Y@     �R@     @Y@     �U@     @T@     �Q@     @P@      N@     �N@      J@      E@     �A@      ;@      8@      :@      9@      5@      6@      5@      *@      8@      *@      $@      &@      *@      *@      .@      $@      @      @      @      @      @      @      @      @      @      @               @              @      �?       @      �?      @      @      @       @              �?               @              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?               @       @       @      �?      �?      @       @      @      �?      @      @      �?      @      @      @      @      &@      @      @      "@      @      &@      @      @       @      ,@      $@       @      "@      2@      4@      7@      9@      7@      A@      <@      <@      ?@      F@      D@     �E@      H@      @@      K@      M@     �I@     �H@     �G@      M@     �I@     �N@      L@      J@     �C@     �I@      C@      C@      8@      2@      B@      A@      >@      1@      <@      2@      8@      *@      ,@      .@      @      @       @      @      @      @      @      @      �?      �?              �?              �?       @       @      �?              �?      �?              �?        ��ub+      o6~	g�zv���A*�V

mean squared errorٻ=

	r-squared@�R>
�>
states*�>	    ��    ��@    ��=A!�d�����)`2��$��@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,�4��evk'���o�kJ%���-�z�!�%����Łt�=	���R�����J��#���j�y�+pm��mm7&c��`���nx6�X� ��f׽r���f;H�\Q������%���K��󽉊-��J��Qu�R"�PæҭUݽ���X>ؽ��
"
ֽ�EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ�_�H�}��������嚽��s���-���q�        �-���q=���_���=!���)_�=|_�@V5�=<QGEԬ=�
6����=K?�\���=;3����=(�+y�6�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >��f��p>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              �?      @      @      $@      *@     �E@      Q@      \@     `s@     x�@     $�@     ��@     ��@     >�@     �@     л@    �9�@    �?�@     ��@     ��@     �@     i�@    �_�@     �@     ��@    �P�@    ���@     ��@    ���@     ��@    ���@    ���@     ��@    �~�@    @&�@    @`�@    ���@    ��@    @{�@    �|�@    ��@     �@    ���@    @`�@     9�@    ���@    ���@    �*�@     ��@    ��@    @V�@    ���@    ���@    ���@    @��@    ��@    @I�@    @J�@    �%�@    @��@    ��@     �@    ���@     9�@    ���@    �r�@     ��@     ��@    ���@     ��@     �@     ع@     ��@     �@     L�@     ȱ@     O�@     Ԯ@     ��@     p�@     <�@     ��@     f�@     ��@      �@     ��@     ��@     ��@     ��@     h�@     h�@      �@     �@     ��@     ��@     ؇@     @�@     ��@     ��@     ��@     h�@      {@     �z@      y@     x@     �s@     `r@     @r@     @p@     �n@     �k@     �i@     �f@     `b@     �d@     �_@     �`@     �a@      ^@     �U@     �Y@     �R@     �U@      U@     @S@     @Q@     @P@      R@     �H@      G@     �J@     �@@     �I@     �F@     �B@     �D@      B@      =@      ?@      6@      :@      7@      9@      5@      8@      *@      0@      (@      .@      *@      1@      $@      1@      ,@      @      &@      "@      @      &@      "@      &@      *@      @      $@      @      @      @      @      @      @       @      @      @      @      @      @      @       @      @      @      @      @       @      @       @      @      @       @      �?       @              @       @       @       @      �?              @      @       @      �?              �?       @              �?               @      �?               @      �?       @      �?      �?      �?              �?              @              @              �?               @              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?              �?      �?              �?              �?              �?               @              �?               @      �?              �?      �?              �?       @              �?      �?              �?              �?      @      �?              �?      �?      �?              �?               @       @              �?              �?      �?              @      @      @       @              �?      @      �?       @      @      @       @      @      �?      �?      @      @      @      @       @              �?       @       @      @      @      @      @      @      @      @      @       @      (@      @      @              @      @      "@       @       @      &@      @      &@      @      $@      &@      &@      2@      4@      .@      *@      ,@      4@      4@      4@      7@      @@      4@      9@      7@      >@      ;@     �B@      A@     �@@      H@      @@      B@     �I@     �F@      H@     �K@     �M@      T@     �R@      T@     �W@     �T@     �X@     @V@     @[@     @]@      c@      c@     `e@     �f@     �d@      l@      j@     �n@      o@     �q@     @s@     `v@     �u@     �x@     {@     `~@     p�@     ��@     �@     8�@     ��@      �@     ��@     x�@     ��@     l�@     ��@     ؖ@     \�@     ��@     ��@     L�@     �@     ��@     ��@     `�@     �@     ګ@     �@     ΰ@     ɲ@     ��@     ��@     ��@     %�@     D�@    �2�@    ���@     �@    ���@     ��@     )�@     ��@     }�@      �@    �9�@    �$�@      �@    ��@    ���@    ���@    �9�@    ���@     ��@    @��@    �9�@    ���@    @��@    @��@    @��@    ���@    ���@    �3�@    @��@     ��@    @��@    �I�@    @��@    �w�@    ��@    @d�@    �"�@    �:�@     ��@    ���@     ��@     �@     y�@     ��@    ���@    ���@     a�@     ��@    ���@     ��@    ��@    ���@     ��@     ��@     D�@     ��@     n�@     8�@     �@     ��@     `�@      j@      Y@     �H@      :@      2@      &@       @      @               @        
�
predictions*�	   `�tѿ    �<
@     ί@!  ���%@)���)2D@2�_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���T7����5�i}1�f�ʜ�7
������1��a˲���[��})�l a�>pz�w�7�>��[�?1��a˲?����?f�ʜ�7
?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?3?��|�?�E̟���?S�Fi��?ܔ�.�u�?�Š)U	@u�rʭ�@�������:�               @              �?               @      @      @      (@      "@     �@@      D@      @@      R@     @S@      U@     @Y@     �Z@     �\@      ]@     ``@     �V@     �W@      X@     @V@     @T@     �Q@      N@     �Q@     @S@      L@      A@     �I@      C@      C@      ;@      :@      0@      9@      C@      9@      6@      &@      .@      $@      @      $@      &@      &@      $@       @       @      &@      @      @      @      &@      @      @      @       @       @      @       @      @      �?      @       @      �?       @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @      @      @      @      @      �?       @      @      @      @      @      @      *@      $@      &@      &@      $@      ,@      &@      1@      *@      (@      8@      <@      >@      2@      :@      :@     �@@      C@      F@      B@      E@     �E@     �E@      N@      K@      I@     �E@     �N@      Q@      M@     �H@     @P@      C@      P@      J@      J@     �L@     �D@     �E@     �G@     �B@      C@      @@      <@      :@      0@      3@      $@      8@      *@      &@      (@      @      @      @      @      @      @      @      @      @              �?       @      �?      �?              �?              �?              �?        *��,      9�!�	��v���A*�X

mean squared error*�=

	r-squared�
q>
�>
states*�>	   @�
�   ��I@    ��=A!�OK3u���)�b^�>��@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�:�AC)8g�cR�k�e�w&���qa�d�V�_���u}��\�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,���-�z�!�%�����i
�k���f��p�Łt�=	���R�����J��#���j�Z�TA[�����"��mm7&c��`���nx6�X� ��tO����f;H�\Q���K��󽉊-��J���1���=��]����/�4����
"
ֽ�|86	Խ;3���н��.4Nν�!p/�^˽�d7���Ƚ��@�桽�>�i�E��G-ֺ�І�̴�L�����/k��ڂ��-���q�        �-���q=V���Ұ�=y�訥=|_�@V5�=<QGEԬ=G�L��=5%���=�Bb�!�=�d7����=�!p/�^�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��==��]���=��1���=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>�J>Łt�=	>��f��p>��-�z�!>4�e|�Z#>4��evk'>���<�)>�'v�V,>7'_��+/>6NK��2>�so쩾4>�z��6>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              �?      @      @      $@      6@      C@      O@     @a@     �t@     8�@     ̕@     ��@     ��@     ��@     	�@     k�@     ſ@    ��@     L�@     ��@     �@     ��@     ��@     3�@    �O�@    ���@     ��@     F�@    �c�@     ��@    ���@     ��@     ��@     i�@    �*�@     ��@    �G�@     ��@    �b�@    @��@     P�@    @��@     ��@     �@    �p�@    @��@    ���@    ��@    �=�@    �u�@    @{�@     ��@     ~�@    ���@    ���@    ��@    ���@    @�@    @�@    ���@     ��@     o�@    �u�@    ���@    ��@    ���@     "�@     \�@    ���@     �@     m�@     r�@     �@     /�@     B�@     o�@     �@     z�@     ��@     �@     <�@     �@     أ@     �@     ��@     D�@     X�@     l�@     ��@     ԕ@     ؒ@     0�@     ��@     ؋@     ��@     Ȉ@     �@     ��@     ؃@     ��@     ~@     @~@     0z@     �y@     @t@      v@     `t@     �q@     `o@     @m@      j@     �i@      f@     �d@     �e@     �a@     �`@     `b@     @[@     �Z@     �\@     @U@     �V@     @R@     @X@     @P@     @Q@     �P@      M@     �L@     �K@     �G@     �C@      H@      D@      =@     �F@      A@      <@      @@      ?@      <@      8@      0@      ;@      2@      3@      5@      .@      "@      .@      0@      6@      *@      @      @      @      "@      "@      $@      "@      @      $@      @      @      "@      @      @      "@      @      @      @      @      @      @      @      @      @       @       @       @       @      �?      @              @       @              @              @       @              �?       @       @      @      �?       @               @      @      �?      �?      �?      �?      �?       @       @      �?               @      �?              �?              �?               @              �?               @              �?      �?               @              �?              �?      �?              �?              �?              �?              �?              �?      �?               @      �?              �?              �?              �?      �?              �?              �?               @      �?               @              �?               @              �?               @      �?               @              �?              �?              �?      �?      �?              �?      �?               @      �?      �?      �?              @              @              �?       @      �?       @      �?       @       @      @               @       @      �?       @      @      @      @      @       @       @      @      @       @      @      @      �?       @      "@       @      @      @      @      @      @      @      "@      @      @      @      "@      @      (@      @      $@      &@      "@      @      ,@      $@      .@      ,@      2@      *@      "@      2@      $@      5@      .@      5@      1@      6@      ,@      7@      <@      3@      E@     �B@     �A@      C@     �A@     �I@      E@     �J@     @P@     �K@      O@     �S@     @T@      T@     @W@     �[@     @Y@     @[@     `c@      `@      c@     @d@     @g@      h@     �j@     �k@     �n@     @q@     `q@     0s@     �u@     �x@      x@     |@     ��@     �@     8�@     Ȅ@     ��@     P�@     @�@     ��@     ��@     ��@     �@     @�@     ��@     ܙ@     �@     h�@     ��@     ��@     P�@     ��@     �@     P�@     "�@     i�@     m�@     ��@     ��@     �@     չ@     ��@     о@    ���@     R�@     #�@     ��@    �6�@     ��@     ��@    ��@    @=�@    �&�@     &�@    �=�@    @D�@     /�@    @��@     ��@    @��@     ��@     R�@    @��@    �\�@    @�@    @��@    @�@    ���@     
�@    ���@    �C�@    @��@    ���@    @��@     �@    �|�@     h�@     ��@    �p�@     ��@     �@     ��@     M�@     ��@     &�@    ��@     ��@    ���@     w�@      �@     ��@    �f�@    �9�@     ��@     ��@     ?�@     F�@     j�@     ò@     ث@     @�@     �@     p�@     �m@      ^@     �R@     �D@      4@      *@       @      @      @       @        
�
predictions*�	   `��ٿ   `�@     ί@!  �\X*�);jX�{�C@2�W�i�bۿ�^��h�ؿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�+A�F�&�U�4@@�$��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r�x?�x����[���FF�G ���Zr[v��I��P=��pz�w�7���f����>��(���>6�]��?����?f�ʜ�7
?>h�'�?�vV�R9?��ڋ?�.�?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?u�rʭ�@�DK��@�������:�              �?               @      �?      @      @      @      @      @      $@      .@      0@      <@     �@@      H@      M@     �Q@     �T@     @[@     �[@     �^@     �a@     �^@     @_@      ^@     �^@     @[@     �W@     �T@      U@     �Q@     @S@      M@     �N@      G@     �C@     �C@      D@     �B@      <@      8@      9@     �A@      9@      2@      *@      4@      @      @       @      @      "@      "@      "@       @       @      @      @      @      @      @       @      @       @      @      �?       @      �?       @      �?      �?      @      @              �?              �?              �?              �?              �?      �?              �?              �?       @              �?              �?              �?              �?      �?              �?              �?              �?      �?              �?      @      �?      �?       @      �?      @       @      @      �?      @       @       @      �?      @      @      @       @      1@      @      &@      "@      "@      @      &@      5@      2@      2@      <@      7@      4@      ;@      ?@      8@      ;@     �C@     �B@      ?@     �G@      B@      H@     �E@     �F@      E@      E@      K@      I@     �F@     �I@     �D@     �G@      I@     �B@      @@      D@     �D@      >@      8@      >@      ;@      6@      ;@      6@      *@      (@      &@      $@      @      @      "@      @      @      @      �?      @      @       @      �?              �?               @      �?       @       @              �?              �?        d���R+      ��k�	l��v���A*�V

mean squared error�=

	r-squaredh^>
�=
states*�=	    ���    ,@    ��=A!X:��B���)��lq.�@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%������f��p�Łt�=	���R����Z�TA[�����"�RT��+��y�+pm�=��]����/�4��PæҭUݽH����ڽ�d7���Ƚ��؜�ƽ�
6������Bb�!澽���6���Į#����-���q�        �-���q=V���Ұ�=y�訥=���6�=G�L��=�
6����=K?�\���=�b1��=��؜��=��.4N�=;3����=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����='j��p�=��-��J�=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>���">�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��o�kJ%>4��evk'>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>/�p`B>�`�}6D>��Ő�;F>��8"uH>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�               @       @      (@      *@      &@      @@      Q@     �Z@      m@     H�@     ��@     6�@     ,�@     ��@     y�@     p�@    ��@    �.�@     ��@     b�@    ���@     ��@    ��@    �\�@     ��@     ��@    ��@     ��@    ���@    ���@     ��@    �k�@    ���@     ��@     �@    ��@     y�@    �e�@     �@    @[�@    @ �@    �W�@    @��@    ���@    ���@    ���@    ���@    @��@    �I�@     ��@    @��@    ���@    �1�@     {�@    ���@    �s�@     ��@    ���@     �@    ���@    ���@     �@     ��@    �k�@     q�@    �w�@     ��@     ��@     ��@     ��@     .�@     �@     )�@     ��@     s�@     ��@     �@     ��@     <�@     �@     Φ@     R�@     .�@     J�@     `�@     ��@     ��@     X�@     h�@     ��@     В@     T�@     0�@     P�@     ��@     �@     ��@     ��@     Ȃ@     �@     �@     �|@     @{@     �y@     pu@     0v@      r@     �r@     �m@      p@     �j@     �k@     �h@     �f@      c@     `d@     �`@     @_@     �^@     �[@     �\@      X@     @R@     �R@     �S@     @V@     @V@     �J@     �H@     �L@      H@     �C@     �E@      =@     �C@      D@      C@      <@      4@      7@      ;@      ?@      8@      2@      4@      4@      &@      1@      2@      ,@      1@      4@      3@      &@      "@      0@      "@      @      @      (@      @       @      &@      &@      $@      @      @       @      @      @      @      "@      @       @      @      @      @      �?      @      �?      @      @      �?      @      @      @      @      @              �?      @      @      �?       @      �?              �?      @      �?              @       @      �?              �?      �?      �?      �?      �?      �?      �?              �?      @       @              �?       @      �?      �?      �?              @      �?              �?      �?      �?              �?              �?              �?              �?              �?              @      @              �?              �?              �?               @              �?              �?              �?      �?      �?              �?              �?              �?              �?              �?      �?      �?              �?               @              �?              �?               @       @       @               @              �?      �?      �?               @      �?      @       @      �?       @       @              @      @      �?      �?      �?      @       @               @      �?      @       @      @              @      �?      @       @      @      @       @      @      @       @       @      @      @       @      @       @      @      @      @      &@      @       @      *@      (@      $@      (@       @      @      $@      "@      (@      0@      5@      *@      4@      .@      3@      3@      1@      4@      :@     �@@      ;@     �D@      8@      A@      @@     �B@     �G@      F@     �E@      D@     �L@      K@     �R@     @P@      U@     �T@     @W@     �X@      [@     @`@     �\@      `@     `a@     �c@     �d@     �g@      k@      l@     `o@     �q@     �q@     ps@     �u@     v@     �w@      }@     �}@     x�@     X�@      �@     ��@     ��@     @�@     ��@     ��@     `�@     <�@     T�@     ؖ@     ��@     8�@     h�@     Ġ@     �@     z�@     (�@     l�@     <�@     ��@     ֯@     ��@     A�@     ʴ@     E�@     ��@     �@     +�@    ��@    �u�@    ���@    ���@     &�@    ���@     O�@    �b�@    �\�@     k�@    @f�@    ���@    @a�@    @4�@    @��@     ��@    @;�@    �X�@     �@    @�@    ���@    �>�@    ���@     R�@     ��@    �/�@    @�@    �>�@    �\�@     ��@    @]�@    @��@    ���@    �3�@     %�@     E�@    ��@     }�@    �G�@    ���@    ���@    ���@    �)�@    ���@    ���@     H�@    �t�@     ��@    ��@     ��@    �)�@     ��@     c�@     i�@     ��@     d�@     ~�@     ��@     �@     ��@     �u@     `g@      [@      B@      8@      7@      "@      @      @      @        
�
predictions*�	   ��׿   � ;
@     ί@!  p\��)̙G��4A@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�+A�F�&�U�4@@�$���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��a�Ϭ(�>8K�ߝ�>>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�P�1���?3?��|�?�Š)U	@u�rʭ�@�������:�              �?              �?              �?      �?      �?      @      @      @      @      "@      4@      >@      =@      B@      C@      E@     �F@     �R@     �V@     �W@     �Z@      _@      `@     �b@     �_@     �a@     �^@     �Z@     �Z@     �S@     �T@     �U@     @T@      O@     �K@     �K@     �H@      G@     �C@     �D@      A@      >@      2@      5@      5@      2@      (@      0@      3@      @      @      0@      "@      &@       @      @      @       @      @      @      @      @      �?      @              @      �?      �?      �?       @      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?      �?              @               @              @              @       @              �?      @      �?      @       @       @      @      @      @      "@      "@      @      @       @      @      .@      @      2@      "@      1@      5@      1@      .@      1@      9@      <@      1@     �A@      6@      9@      E@      9@      >@      B@     �A@      C@      ?@     �C@      G@      C@      F@      I@      E@     �E@      F@      F@      9@     �B@      9@     �B@      ?@      E@      :@      9@      9@      6@      0@      0@      0@      (@      ,@      "@      "@              "@       @      @      @      @       @      @      @      �?      @      �?               @              �?        z��,      ;���	�^�v���A*�Y

mean squared error��=

	r-squared,�h>
�@
states*�@	    Z�   ���@    ��=A!�<����)n���@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,���o�kJ%�4�e|�Z#���-�z�!�%������f��p�Łt�=	���R����2!K�R���J����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%��=��]����/�4���Qu�R"�PæҭUݽH����ڽ���X>ؽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6�����G�L������6������z5��!���)_���-���q�        �-���q=z����Ys=:[D[Iu=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=�8�4L��=�EDPq�=����/�=�Į#��=���6�=��؜��=�d7����=(�+y�6�=�|86	�=���X>�=H�����==��]���=��1���='j��p�=��-��J�=�tO���=�f׽r��=nx6�X� >y�+pm>RT��+�>���">��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              @      @      (@      &@      ;@      ;@     �G@     �\@     `l@     x�@     t�@     ٰ@     �@     G�@     ޺@     ξ@    ���@    ���@     ��@    �!�@    ���@    ���@    ���@     ��@     ��@     ��@    ���@     ��@     i�@    ���@     ��@     �@     ��@    ���@     ��@     ��@    �"�@     ��@     ��@    @m�@    ���@    ��@    @�@    ���@     ��@    ��@    ���@    ���@    ���@     8�@    �`�@    �}�@    ���@    @��@     ��@    ���@    �|�@    ���@    �/�@    @Q�@    @.�@    ���@    �:�@    �)�@    �\�@    �h�@    �B�@    ���@     ��@     =�@     �@     J�@     t�@     +�@     ��@     �@     E�@     6�@     |�@     ��@     ƨ@     ,�@     ��@     X�@     &�@     4�@     d�@     �@     t�@     8�@      �@     ��@     ��@     ��@     ��@     ȋ@     8�@     ��@     h�@     ��@     `~@     �~@     �z@     �x@      w@     v@      t@     �s@     �o@      l@     �k@      k@     `j@     �f@      b@      c@     @c@      b@      _@      ]@      _@      W@     �V@     @V@     �U@     �O@      Q@     �K@      S@      I@      M@     �G@      K@     �D@     �D@      A@      D@      ;@      ?@      @@      <@      >@      2@      ?@      9@      3@      *@      2@      2@      .@      *@      .@      ,@      ,@      *@      (@      &@      0@       @      @      @      *@      @      @      "@       @      "@      @       @       @      $@      @       @      @      @      @       @      @      @      @       @      @      @       @      �?       @      @      @      �?               @      @              @              @       @      �?      �?              �?       @      @      �?       @      �?       @      �?      �?      �?      �?      �?      �?       @      �?              @      �?      �?              �?      �?              �?              �?      �?              �?              �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?       @      �?              �?              �?               @      @              �?              �?              �?              �?              �?      �?              �?               @               @              �?      @      �?              �?      �?              �?      �?               @      �?      �?      �?              �?       @      �?      �?      �?       @      @              @              �?      �?               @              �?      �?       @      �?      �?              @      �?      �?      @       @      �?      �?       @      @      �?      �?       @               @      @      @      �?      @      @       @      �?      @       @      @       @      @      @       @      @       @      @      @      @      @       @      @      @      @      "@      &@      *@      @      $@      .@      @      &@      1@      .@      $@      .@      ,@      0@      2@      $@      4@      6@      1@      >@      7@      4@      5@      :@      E@      :@      H@     �H@      J@      J@      D@      G@     �R@     �N@     @T@     @T@     �Q@     @T@     @V@     �\@     �\@     �_@     �_@     ``@     @d@     `g@     @h@     `g@     �l@     �n@      q@      o@      t@     �t@     Pu@     �y@     �|@      ~@     ��@     H�@     `�@     0�@     H�@     ��@     ��@     X�@     <�@     T�@     �@     ��@     ��@     �@     ě@     ��@     ��@     p�@     V�@     �@     �@     @�@     ��@     ��@     ұ@     ӳ@     ��@     ��@     A�@     ��@     Ͽ@    ��@     ��@    �O�@     �@    ���@    ���@     ��@     D�@    ���@    ��@    ��@    �*�@     ��@    ���@    @"�@    @V�@     ��@    @��@    @g�@    @w�@    �J�@    @��@    �H�@    @��@    ���@     4�@     �@    ���@     ��@    �#�@    @��@    @y�@    @	�@    ���@    @��@     �@     a�@     ��@    ���@    �u�@    ���@    ���@    ���@    �M�@    ���@    ���@    ���@    ���@     ��@    ���@     ��@     ��@     ��@     Թ@     Y�@     ڳ@     �@     l�@     F�@     x�@      s@      r@     �d@      Z@      F@      5@      &@      (@      @       @        
�
predictions*�	   @JԿ   �zg
@     ί@!  8C��)@)�/ۢC@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�>h�'��f�ʜ�7
���[���FF�G �})�l a�>pz�w�7�>>�?�s��>�FF�G ?��[�?�5�i}1?�T7��?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?S�Fi��?ܔ�.�u�?��tM@�Š)U	@u�rʭ�@�������:�              �?      �?              �?      @      @      "@      (@      @      2@      ,@      ;@      =@     �B@     �I@     �L@      O@     �U@      Q@      U@     @W@      Y@     �Y@     �T@     @W@     @Y@     @V@     �T@     @U@     �R@     �Q@      L@      P@      R@      L@     �@@     �B@     �F@      C@      <@      =@      <@      8@      $@      2@      =@      0@      (@      2@       @      $@      @      $@      @      "@      @      @      @      @      @      @      @       @      @       @      @      �?      @      @       @      @       @      �?       @              �?      �?              �?              �?       @               @              �?              �?              �?              �?      �?              �?              �?              �?       @       @      @      �?       @       @      �?      �?      �?      �?              �?              @      @      @      @      �?      @      @      @      @      "@      $@       @      &@      @      (@      ,@      .@      *@      1@      2@      0@      1@      ;@      3@      7@     �@@      E@      ;@      @@     �H@      J@     �F@     �I@     �K@      I@      K@      L@     �R@      E@      O@      K@      K@     �H@      I@     �G@     �F@     �K@      N@      E@      E@      C@      4@      0@      =@      0@      ,@      9@      *@      1@      .@      $@      @      @      *@       @      "@      @       @              @      @               @      @      �?       @              �?      �?              �?        ��-      c,�`	Q�#w���A*�Y

mean squared error�:=

	r-squared��t>
�A
states*�A	    ��   �2o@    ��=A! �Mz��)���S�@2� h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%���-�z�!�%�����i
�k���f��p��#���j�Z�TA[���`���nx6�X� ��f׽r����tO����f;H�\Q������%��'j��p���1����/�4��ݟ��uy�z�����PæҭUݽH����ڽ���X>ؽ��
"
ֽ�d7���Ƚ��؜�ƽ�
6������Bb�!澽G�L������6���Į#�����M�eӧ�y�訥�V���Ұ����@�桽�����嚽��s��\��$��%�f*��8ŜU|�x�_��y�:[D[Iu�z����Ys��-���q�        �-���q=���_���=!���)_�=�Į#��=���6�=�b1��=��؜��=��.4N�=;3����=(�+y�6�=�|86	�=���X>�=H�����=�Qu�R"�=i@4[��=z�����=ݟ��uy�='j��p�=��-��J�=�K���=�9�e��=����%�=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�               �?      @      @      (@      5@      D@      L@      ^@      p@     @�@     �@     {�@     ��@     ��@     W�@     �@    � �@    �v�@     (�@    ���@     n�@    ���@     �@    ���@     h�@     ��@    ���@     �@    ���@     ��@     ��@    �R�@     ��@     �@    �$�@     ��@    ��@    �o�@     �@    ���@    @��@    @��@    ���@    @��@    @i�@    ���@    ���@    ���@    �C�@    �q�@    �u�@    �c�@    ���@    @��@    ���@    ��@    @D�@    ���@    ��@    @�@     #�@    ���@    �p�@     H�@    ��@     ��@    ��@    ��@     ��@    �	�@     ��@     �@     V�@     $�@     ε@     w�@     �@     �@     <�@     �@     ��@     ��@     ��@     ^�@     d�@     ��@     0�@     �@     �@     �@     ��@     Ԓ@     Đ@     Ȏ@     ��@     H�@     Ј@     0�@     ��@     x�@     Ȁ@     �~@     P}@     �y@     0w@     �t@     �t@      r@     �p@      t@     �l@     �j@     �i@      f@     `f@     @c@      e@     �b@     @\@      Z@     �Y@      [@     �]@      Z@     �T@     @T@      R@     �P@      K@     �N@      J@      Q@     �D@      J@     �E@     �D@      @@     �D@      9@      3@      5@      :@      3@      1@      5@      9@      6@      ,@      @      ,@      *@      .@      .@      ,@      "@      &@      @      2@      $@      @      (@      $@      @       @      @      "@      @      @      @      @      @      @      @      @      @       @       @      @      @      @      �?      @      @      @       @      @       @              @       @       @      @      �?      �?      @              �?              �?              �?       @       @               @      �?      @      �?      @      �?              @              �?      �?              �?              �?      �?      �?              �?               @              �?               @              �?              �?      �?              �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      @      �?      �?              �?              �?       @              �?              �?       @              �?      �?      �?      �?              �?       @              �?              �?              @      �?              �?       @              �?              �?      @      @              �?      �?      @              @      �?      �?              �?      @      @              @      @      @      @      @       @      @      @      @      @      @      �?      @      @      @      @      @       @      @      @              @      @      $@      @      @      @      "@      $@      &@       @      @      .@      "@      .@      0@      2@      (@      (@      .@      .@      2@      2@      7@      6@      3@      3@      :@      9@      A@      A@     �C@      @@     �B@     �B@      D@     �A@      O@     �L@     �L@      O@     �N@     �T@      T@      T@     @Y@      [@     �]@      Z@     �[@     `b@     `d@     �c@     @d@     �d@     @j@     `i@     �j@     0q@     �r@     �q@     �v@      w@     0y@     �{@      ~@      �@     x�@     ��@     �@     x�@     0�@     ��@     x�@     �@     ��@     t�@     ��@     ȗ@     Ě@     8�@     p�@     ��@     x�@     ��@     `�@     ��@     @�@     ��@     ް@     y�@     v�@     ��@     C�@     n�@      �@     
�@    �>�@     e�@    ���@    ��@     ��@     ��@    ���@     ��@     ��@     ��@    ���@     ��@    ���@    �H�@    @ �@    @e�@    ���@    ���@    ���@    �m�@    ���@    @��@    �)�@     $�@     ��@    @7�@     ��@    @��@     ��@    ���@    @��@    �H�@    ���@     ~�@     ��@    ���@     ��@    ���@     ��@    �u�@     $�@     ��@    �%�@     ��@    ��@    ��@     u�@    �^�@    �@�@    �l�@    ���@     ��@     X�@     Ĺ@     2�@     ��@     M�@     4�@     H�@     ��@     �t@     �n@     �m@     @b@     �T@      (@      &@      *@      @      �?      �?        
�
predictions*�	   ��$�   ��-@     ί@!  P���-@)�W��5C@2��1%���Z%�޿�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+���ڋ��vV�R9��T7����ѩ�-߾E��a�Wܾ����?f�ʜ�7
?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?u�rʭ�@�DK��@�������:�              �?              �?               @              �?      �?       @      *@      @       @      2@      <@      A@      >@     �I@     �E@      R@     @R@     @S@     �S@      V@     @V@      Y@      [@     @_@     �[@      Y@     �Z@     @U@     @S@     @U@     �N@     @Q@     �P@     �J@      E@     �B@     �E@     �A@      A@      :@      >@      9@      $@      4@      "@      *@      .@      "@      (@      "@      .@      *@       @       @       @      @      @      @      @      @       @      @      �?      @      �?      �?      @      @      @      @      �?              �?       @      �?              �?      �?              �?              �?              �?              �?              �?              �?       @               @       @       @      @      @      @      �?       @               @       @      @      @      @      @      @      @      @      @       @      @      @       @      $@      $@      @      1@      &@      "@      @      3@      .@      6@      8@      .@      <@      ?@      7@     �B@     �A@     �B@      I@      J@      K@     �P@     �J@     �H@     �H@     �N@      J@     �F@     �O@     �G@      E@      N@     �I@     �F@     �F@     �B@      I@     �A@     �@@      ?@      <@     �A@      7@      5@      7@      2@      ,@      $@      "@      @       @      @      @      @      @               @       @               @              @      �?      �?              �?              �?              �?        K��b,      d�$]	��Nw���A*�X

mean squared errorN�=

	r-squared�w>
�?
states*�?	   �V��   @z@    ��=A!d��,��)���d�@2�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�������M�6��>?�J���8"uH��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	��J��#���j�Z�TA[��RT��+��y�+pm��`���nx6�X� ��f׽r����tO����f;H�\Q������%���Qu�R"�PæҭUݽH����ڽ���X>ؽ�!p/�^˽�d7���ȽK?�\��½�
6������EDPq���8�4L���|_�@V5����M�eӧ���@�桽�>�i�E��e���]����x�����-���q�        �-���q=e���]�=���_���=�8�4L��=�EDPq�=����/�=�Į#��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=��1���='j��p�=��-��J�=f;H�\Q�=�tO���=�`��>�mm7&c>y�+pm>RT��+�>Z�TA[�>�#���j>��R���>Łt�=	>%���>��-�z�!>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�              �?       @      @      *@      =@      C@     �Q@     �`@     �q@     8�@     �@     =�@     I�@     �@     D�@     ��@    �g�@     ��@     ��@    ���@    �3�@     �@    ���@    ���@    ���@     w�@     ��@     ��@    ���@     ��@    ���@    ���@    ���@    ���@     U�@     3�@    �h�@    ��@    @%�@    @��@    @ �@    ���@    @X�@    ���@     ��@    �?�@    �i�@    �~�@    �Q�@    @�@     �@     ��@    ���@    ��@     a�@     ��@    ���@    @��@     ��@    ���@     ��@    �f�@     r�@     �@    ���@     b�@     v�@    ���@    �E�@     +�@     
�@     ��@     {�@     ��@     Ų@     R�@     T�@     �@     �@     §@     V�@     �@     ��@     �@     �@     8�@     p�@     ԗ@     ��@     ��@     (�@     ��@     ��@     ��@     ��@     ��@     �@     h�@     `�@     0�@     0~@     �}@     Pz@      y@     �v@     w@     Pq@     pq@      r@      l@      j@      j@     �h@     �g@     �d@     �c@     �a@      `@     @\@      ]@     �[@      `@     �Y@     �R@      U@     �S@     �N@      R@      M@      P@      P@      F@     �D@     �H@     �C@     �A@     �J@      :@      @@      <@      =@      <@      7@      &@      6@      2@      5@      4@      (@      ,@      2@      *@      $@      2@      .@      ,@      (@      *@      @      &@      @      @       @      "@      @      @      @       @      @      @      @      @       @      @              �?      @      @      @       @      @      @      @      @      @       @      @      �?      @       @      �?      @       @      �?       @       @       @       @       @              @       @              �?      �?       @               @      �?              �?              �?      @      �?               @       @              �?       @      �?              �?      @              �?              �?              �?              �?              �?              �?              �?               @              �?              �?              �?              �?              @      @              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?              �?              �?      �?              �?              �?       @       @              @              @               @              �?              �?       @       @      �?       @              �?      �?              �?              �?              @      @       @      @       @              �?      �?      �?       @      @      �?      @      �?      �?              �?      @       @      @      @       @      @      @      �?      @      @      @      @      @      �?      �?      @      @      @      $@      @       @      @      @      ,@      @      &@      @      @      $@      @      ,@      "@      (@      0@      ,@      0@      4@      (@      ,@      1@      1@      .@      7@      8@      :@      6@      ;@      =@      C@      A@      C@      ?@      @@      A@      H@      I@     @Q@      O@      Q@     �O@     �W@      R@     �V@     �X@     �V@      ^@     �`@     ``@      b@     @`@     @d@     �d@     @g@     @l@      l@     �k@     �p@     �r@     �p@     s@     �v@     �w@     `{@     �}@     ��@     �@     `�@     ��@     �@     H�@     h�@     �@     �@     ��@     t�@     `�@     \�@     ��@     ̝@     ��@     ��@     ֢@     ��@     j�@     v�@     ܫ@     P�@     &�@     ��@     e�@     3�@     ��@     ��@     ��@     ʿ@    ���@     R�@    ���@    �3�@    �d�@     ��@     ��@     >�@    �Q�@     �@     ��@    @��@    �d�@    ���@    @E�@    ���@     ��@     ��@    @B�@     ��@    ���@    ���@    ��@    @��@     B�@    ���@     ��@    �X�@     %�@    ���@     ��@    @n�@    ���@    �G�@    �<�@     s�@    �w�@     ��@    ��@     %�@     x�@    �@�@     ��@     ��@     ��@     7�@    �-�@     ��@     ��@     =�@     ��@     ��@     �@     ��@     ��@     ��@     j�@     f�@     ȉ@     �w@     �p@     �l@     �]@      E@      2@      5@      @      @       @        
�
predictions*�	    �׿   �w^
@     ί@!  Ptɤ�)m^�^��D@2��^��h�ؿ��7�ֿ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��f�����uE����I��P=�>��Zr[v�>>h�'�?x?�x�?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?ܔ�.�u�?��tM@�Š)U	@u�rʭ�@�������:�              �?              �?      �?       @      �?       @      @      @      &@      1@      *@      8@      E@      H@      G@     �S@     @U@     @U@     �]@     �_@     @a@      b@      a@     �a@     @]@     �]@     �^@     �\@     �X@      W@      P@      P@      Q@      F@      B@      E@      D@      =@     �B@      =@      9@      4@      8@      0@      0@      2@      *@      *@      (@      "@      "@      *@       @       @      $@      "@       @      @      �?      �?      @       @      @      @      @               @       @       @       @               @      �?              @       @      @      �?               @              �?              �?              �?              �?              �?              �?      �?      �?      �?               @              �?      @       @      �?      �?      @      �?      �?      @       @       @      @               @      @      @      @      $@      $@      @      @      @      @      0@      "@      &@      .@      (@      *@      2@      @      *@      .@      *@      1@      2@      A@      ;@      <@      8@      ;@      @@     �B@      8@      ?@     �E@     �C@     �C@     �@@      D@      A@      G@     �B@      F@     �C@      @@     �D@      B@      ?@      >@     �@@      2@      ;@      :@      4@      :@      2@      1@      6@      2@      0@      "@      @      @      "@       @      "@      @      @      �?      @               @      @      �?      �?              @              �?              �?        ���-      =�	v5xw���A*�[

mean squared errorsD=

	r-squared�Ä>
�B
states*�B	   ��   `�@    ��=A!��j�����)N�O���@2� h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#��i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[��RT��+��y�+pm�nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K�����1���=��]���ݟ��uy�z�����i@4[��PæҭUݽH����ڽ;3���н��.4Nν�!p/�^˽�d7���Ƚ���6���Į#���y�訥�V���Ұ����s�����:��̴�L�����/k��ڂ��-���q�        �-���q=z����Ys=:[D[Iu=������=_�H�}��=<QGEԬ=�8�4L��=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�b1��=��؜��=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=ݟ��uy�=�/�4��==��]���=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>%���>��-�z�!>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�               @      @      @      *@      9@     �A@      T@     `b@      q@     ��@     �@     �@     
�@     �@     J�@     ��@     ��@     C�@      �@    �_�@     ��@     �@     4�@     P�@     �@    ��@    ���@     ��@    �A�@    ���@     e�@     <�@     {�@     ��@    ��@    �O�@    �`�@    �r�@    � �@     ?�@     =�@     ��@    ���@    @��@     e�@     ��@    ���@    ���@    ���@    ��@    �&�@     >�@    @w�@     ��@    ���@    ���@    ���@    @��@    @)�@     (�@     :�@    �@�@    ���@    ���@    ���@    ��@    �M�@     ��@    ���@     ��@     X�@     ɻ@     �@     ��@     ��@     ��@     r�@     ��@     *�@     r�@     ��@     ��@     n�@     <�@     ��@     Ƞ@     L�@     ��@     �@     �@     ܔ@     ��@     ��@     А@     P�@     ��@     0�@     @�@     h�@     H�@     ��@      @     �}@     �y@     �w@     �w@      y@     �r@      s@     �q@     Pp@      k@     �h@     �e@      e@     �d@     �a@     �c@     �`@     �\@     @]@     �Y@     @V@     �Z@     �X@     @V@     �V@     �L@      M@     �I@      S@      G@      M@     �D@      D@      A@      B@      C@      C@      ?@      >@      ?@      >@      7@      0@      ;@      3@      A@      2@      6@      5@      ,@      $@      $@      $@      &@      $@      ,@       @      "@      &@      @      @      "@       @      @       @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?      @      @              @      @      @       @      �?              @      @      @      @       @      �?      �?      �?              �?              @      �?      �?      �?      @      �?      �?      �?      @      �?               @      @      �?              �?              �?       @              �?      @      �?       @              �?      �?              �?              �?              �?      �?              �?               @              �?      �?              �?              �?              �?              �?              �?              �?              �?              @      "@              �?              �?              �?              �?              �?               @              �?              �?      �?              �?      �?              @              �?      �?               @      �?               @      �?              �?      �?      �?              �?      �?              �?      �?      �?       @      @              @              @      �?      �?      @      �?               @      �?      �?              �?       @       @               @      @      �?               @      @      @      @      �?              @      �?               @       @      @       @      @      �?      @      @      @      @               @       @      @       @      "@      @      @      @      @       @      @      $@      @      @      @      @      @      (@      "@      &@      *@      $@      "@      "@      @      0@      0@      0@      3@      (@      1@      ,@      0@      ;@      7@      2@      ;@      8@      6@      7@      9@      >@      :@     �G@      ?@     �D@     �F@     �E@     �G@     �H@      K@      O@     �I@     �P@     @S@      Q@     @U@     @W@     @Z@     @Z@     �X@     @\@      b@     �c@     @g@     @e@     �f@      i@      j@     �k@     �o@     q@     �s@     0t@     �x@     @y@     P~@     �~@     0�@     @�@     0�@      �@     ��@     ��@     �@     ��@     ��@      �@     ��@     ��@     �@     �@     ؞@     ��@     ֡@     ��@     0�@     �@     ª@     �@     �@     \�@     ��@     �@     �@     i�@     l�@     �@    �S�@    ��@     r�@    �@�@     ��@    ��@    ���@     K�@     x�@    ���@    @��@    ���@    @��@    @W�@     <�@     \�@    ���@    @��@     ��@    �<�@     ��@    ���@    �o�@    �$�@    ��@    @��@    @R�@    @6�@     ��@    �Z�@     ��@     ��@    �K�@     �@    ��@     ]�@    ���@     �@     ^�@    ���@     ��@    ���@     ��@     ��@    ���@     ��@     ��@     M�@     ��@     k�@     	�@    �i�@    �e�@     ޽@     _�@     $�@     �@     G�@     ��@     v�@     �@     `~@     �r@      m@     �[@     �E@      9@      1@       @       @      @      �?        
�
predictions*�	    Pi׿   �~s@     ί@!  `Y�@)主^s?D@2��^��h�ؿ��7�ֿ�Ca�G�Կ�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��pz�w�7��})�l a�jqs&\��>��~]�[�>��>M|K�>�_�T�l�>��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?yL�����?S�Fi��?{2�.��@!��v�@�������:�              �?      �?              @      �?      @      @      (@      .@      (@      <@      =@      =@      @@     �E@     �M@      P@     �Q@      V@     �U@     @X@      [@      \@     @Y@     �U@     �\@     @W@     �X@     @V@     �U@     @R@     �P@      J@     �J@      N@      D@      >@      D@     �D@     �A@     �A@     �A@      6@      .@      4@      7@      9@      ,@      0@      &@      *@       @      $@       @      $@       @       @      @      @      @               @      @       @      @       @      �?      �?              �?      �?      �?      �?      �?      @       @       @      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?      �?               @              �?      @              �?       @      @       @       @              �?              �?      @      @      @              @       @      @      @      @      @       @       @      "@      @      @      &@       @      "@      &@      *@      7@      6@      ,@      5@      8@      2@      <@      =@      ?@      ;@      I@      D@      <@      G@      H@     �J@      K@     �E@     �M@      P@     �G@     �J@      I@      F@      J@     �K@     �B@      F@     �J@      ?@     �H@      A@      <@      :@      =@      3@      0@      .@      5@      *@      (@      2@      (@      $@      "@      "@      @      @       @       @      @      �?      �?       @              �?               @       @      �?              �?              �?        Q�j-      ?4�O	�}�w���A *�Z

mean squared errorN�=

	r-squaredxB�>
�A
states*�A	   �G�   @@    ��=A!@�,H���)Z�'���@2� h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J��#���j�Z�TA[��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���/�4��ݟ��uy�z��������X>ؽ��
"
ֽ��.4Nν�!p/�^˽��s�����:������z5��!���)_���-���q�        �-���q=V���Ұ�=y�訥=�EDPq�=����/�=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=�!p/�^�=��.4N�=(�+y�6�=�|86	�=��
"
�=���X>�=PæҭU�=�Qu�R"�=i@4[��=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�                @      @      "@      1@      8@     �I@     @W@     �b@     Pr@     ��@     ��@     ��@     ݳ@     �@     ��@     ʿ@    �h�@     	�@    ���@    ���@     `�@     ��@    ��@    ���@     ��@     ��@    �i�@     .�@     '�@    �I�@    ���@    �<�@    ���@     J�@    �?�@    ���@    �n�@    ���@    �s�@    ���@    @G�@    @��@    �E�@    @��@     ��@     S�@     �@    �A�@    ���@    ���@    �F�@     W�@    @Q�@    �a�@    @��@    @X�@    @��@     ��@    @��@    ��@    ���@    ���@      �@     ��@     ��@    �+�@     ^�@     ��@     ��@     N�@     ɼ@     ��@     R�@     Y�@     '�@     W�@     {�@     ʯ@     d�@     ��@     $�@      �@     ԣ@     ,�@     ��@     ��@     H�@     |�@     �@     ��@     8�@     ��@     А@     ؎@     H�@     `�@     h�@     0�@     0�@     �@     ��@     0~@      y@      y@     �w@     �v@     �t@     Ps@      p@     �o@      n@     �i@     @h@      g@     @e@      d@     �d@     �a@     �]@     �a@      \@     �]@     �T@     �Y@     �V@     �O@      W@     �P@     �G@      R@     �E@      O@      K@     �A@      E@     �D@      =@      C@      >@     �C@      7@      @@      7@      9@      0@      9@      5@      1@      4@      $@      ,@       @      ,@      &@      (@       @      @      "@       @      @      *@      $@      @      "@       @      @      @       @      &@      @      @      @      @      @      @      @       @      @      @      @      @      @      @       @      @       @       @      @      @              @       @      @      @      @              @      @      �?      @      @      �?              �?              �?       @              �?               @              �?              �?       @      �?      �?       @              �?      �?              �?      �?      �?      �?      �?              �?      �?               @              �?      �?              �?       @              @              �?              �?              �?               @      @              �?               @              �?              �?      �?      �?      �?               @              �?               @              �?       @              �?               @              �?              �?              �?      �?      �?      �?              �?      �?              @              �?       @       @      �?      �?              �?              �?              @       @       @      �?              �?      �?       @       @      �?      �?      �?              �?       @      �?      �?      @       @      @      @       @      @      @      �?       @      @      �?      @      @      @      �?      @      @       @      @      @      @      @      @      @      �?      @       @      @       @      @       @      @      @      @      @      @      $@      @      "@      @      &@      *@      $@      1@      (@      .@      ,@      ,@      *@       @      7@      *@      0@      =@      9@      4@      1@      <@      1@      @@      6@      @@      :@     �D@      F@      G@      D@     �H@      G@     �K@     �D@     �N@      O@      R@     �T@      S@     �V@     �Y@      \@     @X@      ^@      a@      b@     �e@      f@      h@     `f@     �j@     �m@     `p@     �q@     �q@     �v@     �v@     z@     `z@     �}@     @�@     8�@     ��@     x�@     h�@     H�@     ��@     �@     P�@     �@     ȓ@     �@     �@      �@     ��@     ��@     ڠ@     j�@     Σ@     h�@     �@     ��@     ��@     )�@     3�@     ��@     ��@     ��@     ��@     ��@     G�@    ���@     '�@    ���@    �M�@     J�@     ��@     ��@    ���@    �3�@    @��@    @��@    ���@    ���@    @c�@    @O�@    �-�@    ���@    �y�@    ��@    @��@    ���@     ~�@     n�@    ��@    @��@    ���@    @a�@    @M�@    ���@     g�@    ��@    ���@    ���@    �l�@     ��@    ���@    ���@    ���@     o�@    ���@    �T�@    ���@    ���@     +�@     ��@     ;�@     0�@     ��@    �o�@     G�@     `�@     S�@     H�@     d�@     k�@     ��@     β@     v�@     ��@     \�@     ��@      w@     �b@     @S@     �C@      6@      $@      @      @      �?      �?        
�
predictions*�	   `�Fӿ   �B�@     ί@!  B-7@)�9�?�H@2��Ca�G�Կ_&A�o�ҿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(���ڋ��vV�R9���d�r�x?�x��>h�'��f�ʜ�7
�.��fc���X$�z����d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?{2�.��@!��v�@�������:�               @              @      @       @      @      @      0@      4@      ;@      4@      C@     �J@      H@      R@     �T@     �Q@      T@     �V@     @W@     @T@     �X@      [@      S@      X@      W@     �S@     �X@     �S@     �P@     �I@      M@     �J@     �@@     �D@      E@      @@      9@      9@      5@      <@      <@      8@      2@      8@      3@      *@      0@      "@      @      &@      @      "@      $@      @      @      @      @      @      @      @      �?       @      @      @      �?      @       @      @              �?      �?       @       @       @      �?      �?              �?              �?              �?              �?              �?      �?              �?               @      �?               @      �?              @              �?      �?      @       @      @              @      �?      @       @      @      �?      @       @      @      @      @      @      @      @      @      @      .@      ,@      *@      0@      (@      0@      9@      5@      5@      7@      ;@      @@      7@      C@      =@     �A@     �B@     �B@      B@     @Q@     �G@      M@     �F@      N@     �I@     �D@      L@      I@     �E@      J@      E@     �@@     �J@     �F@      J@      H@     �E@     �E@      C@     �C@     �A@      :@      <@      8@      4@      1@      .@      @      &@       @       @       @      @      �?      @      �?      @       @      @       @      �?      �?              �?      �?               @              �?        O��ir.      Dl�	���w���A!*�\

mean squared error�M=

	r-squaredhH�>
�B
states*�B	    �V�   @œ@    ��=A!��������)���̧�@2�!h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k�Łt�=	���R�����J��#���j�Z�TA[�����"�y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���K��󽉊-��J�'j��p�H����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽK?�\��½�
6�����5%����G�L���V���Ұ����@�桽���_����e���]���-���q�        �-���q=G-ֺ�І=�1�ͥ�=e���]�=���_���=!���)_�=�>�i�E�=��@��=V���Ұ�=y�訥=<QGEԬ=�8�4L��=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=;3����=(�+y�6�=�|86	�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�!               @      @      @      @      :@     �K@     �Z@     `e@     @t@     ��@     �@     �@     ��@     �@     W�@    ���@     ��@    �{�@    ���@    ���@     c�@    ���@    ���@    ���@    ���@     ��@    ���@    ���@    ���@     z�@    �e�@    �i�@    �@�@     ��@    ��@     ]�@     ��@    ���@     ��@    �U�@    ���@     H�@    @��@     C�@    @��@    ��@     :�@    �_�@    @@�@    @�@    �R�@    ��@    �%�@     d�@     F�@    �H�@     "�@    ���@    @ �@     ��@    � �@     m�@    ���@    ���@    ���@    ��@    ��@     ��@    �*�@     u�@     ��@     ��@     ��@     '�@     L�@     ��@     ��@     z�@     ��@     0�@     t�@     ��@     ��@     |�@     ��@     0�@     �@     X�@     ,�@     (�@     ��@     h�@     �@     ��@     ��@     0�@     ��@     X�@     p�@      �@     ��@     �@     P{@     �{@     �x@     0u@     pt@     @s@     �q@     �r@     @m@     �l@      i@     @j@     �a@     @a@     `a@     @`@     @^@      b@     @Z@     �[@     �W@     @V@      U@      P@     �N@     @P@      T@      Q@      H@     �D@     �C@     �B@     �G@      @@     �C@     �D@      C@      8@     �@@      <@      >@      9@      6@      7@      4@      9@      6@      2@      *@      @      @      $@      $@      $@      "@      @      &@      &@      "@      @      @      &@      @      @      @      @      @      @      @      @      �?      @      @              @      @      @      @       @      @               @      �?      �?      �?      �?      @      @      @              @      �?       @       @      @      �?      @      @      @      �?      �?      @       @              �?       @      �?      �?      @      �?       @      @               @       @      @       @      @       @      �?      �?              �?              @              �?              �?      �?              �?      �?              �?              �?      �?              �?              �?      �?      �?      �?              �?      �?              �?              �?               @              �?              @      @              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              �?      �?      �?      �?              �?      �?      �?      �?              �?               @               @       @      �?      �?              �?               @      �?              �?      �?      �?      �?      @              �?              �?      �?              @      �?      �?              @      @               @               @      @       @              @              @      @       @       @               @              �?      @              @      @      @      @      @       @      @      @      @      @      @      @      @              @       @       @      @      �?      "@      @      @      &@      @      @      @      @      $@      @      @      *@      @       @      &@      @      @      "@      6@      (@      4@      .@      .@      ,@      7@      0@      7@      3@      3@      8@      =@      8@      ?@     �B@      B@      @@     �H@     �C@     �@@      G@      P@      I@      P@      Q@      K@     �R@     �W@     �V@      V@      X@      `@     �_@     �^@     @b@     @e@     �c@     �g@      g@     @i@     `l@     r@     �q@     �v@     �u@     0w@     Pz@     {@     @}@     �@     8�@     ��@     �@     X�@     8�@     h�@     x�@     �@     ��@     ��@     ܗ@     �@     X�@     x�@     &�@     ��@     ��@     �@     �@     4�@     �@     @�@     ��@     m�@     �@     �@     ��@     ��@     n�@     ��@     ��@     ,�@    ���@    ���@     ^�@     D�@    �)�@     (�@     �@    @��@    @��@     ��@    @��@    @i�@    ���@    �@�@    �J�@    �P�@    ���@    @��@    �K�@    � �@    � �@    @��@     ��@     w�@    ��@    ���@     G�@    ��@    @l�@    ���@    �#�@    �Q�@     ��@     t�@     ��@    �J�@    �H�@     ��@    �J�@    �#�@    �/�@    ���@    ���@    ���@    ���@     ��@    ���@    ���@     v�@    ���@     [�@     ˼@     L�@     ��@     S�@     ��@     ��@     T�@     �~@     `m@     �\@      N@      C@      2@      .@      "@      @      @        
�
predictions*�	   ���ѿ   �O@     ί@!  ���)@)�HkG@2�_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��f�ʜ�7
�������[�=�k���*��ڽ�O�ʗ��>>�?�s��>�FF�G ?��[�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?u�rʭ�@�DK��@�������:�              �?      �?      �?      @      @      @      "@      ,@      0@      <@      >@      F@     �J@     �J@      P@      T@     �P@      P@     �R@     �V@     �X@     �Y@      V@     @X@      V@     @Z@      Q@     �S@     �S@     �R@      P@     �O@      L@     �J@      M@      >@      :@      ?@      A@      A@      9@      2@      1@      3@      7@      (@       @      (@      &@      "@      &@       @      @      @      �?      @       @      �?      @      @      @      @      @              @       @      �?       @              �?       @               @      �?               @      @      �?       @      �?              �?               @              �?              �?              �?              �?              �?      �?              �?      �?      @       @      �?       @              @      �?      �?              @       @      @      �?      @      �?      @       @      �?      @      �?      @      @      @      (@      @      @      $@      "@      @      @      &@      (@      1@      5@      2@      5@      6@      7@      ;@      @@      A@      <@      C@     �C@     �A@     �H@      G@      J@      H@     �P@      K@     �G@      I@     �K@     �J@     �E@      J@      M@      I@     �E@     �H@     �A@      F@      E@     �F@     �A@      6@     �@@      :@      7@      6@      3@      1@      4@      (@      (@      @      "@      @      @       @      "@       @      @      @      @      �?      @              @      �?      �?              @              �?              �?              �?        ��Ŝ".      YgB	w��w���A"*�\

mean squared error�=

	r-squaredɊ>
�B
states*�B	   ���   ���@    ��=A!<2!E���)s������@2� h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,�4��evk'���o�kJ%�4�e|�Z#���-�z�!��i
�k���f��p�Łt�=	���R����2!K�R���#���j�Z�TA[��RT��+��y�+pm��mm7&c�nx6�X� ��f׽r����tO�����9�e����K�����1���=��]���ݟ��uy�z������|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�b1�ĽK?�\��½�
6������Bb�!澽�Į#�������/����@�桽�>�i�E��!���)_�����_����G-ֺ�І�̴�L�����8ŜU|�x�_��y�z����Ys��-���q�        �-���q='1˅Jjw=x�_��y=\��$�=�/k��ڂ=��M�eӧ=|_�@V5�=����/�=�Į#��=���6�=G�L��=K?�\���=�b1��=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�               @      @      @      3@     �A@      I@      [@     �c@     �w@     `�@     �@     ܶ@     ��@     ��@     ٽ@    ���@    �o�@     �@     0�@    ���@    �8�@     ��@     ��@     \�@     g�@     �@    ���@    �`�@     ��@    ���@    ���@     ��@     �@    ��@     ��@    ��@     ��@     ��@    �#�@    @��@     �@    �m�@    @E�@     ��@    �J�@    �W�@    �U�@    �H�@    ���@    @�@    @�@    ���@     1�@    �k�@    @��@    �^�@    @��@    ���@    @��@     ��@    �o�@    �,�@    �'�@    ���@    �'�@     ��@     w�@     �@    ���@     ��@     �@     ��@     ��@     �@     ɳ@     �@     �@     D�@     ��@     >�@     �@     �@     ��@     ��@     ܠ@     d�@     �@     `�@     ĕ@     h�@     ��@      �@     <�@     ��@     0�@     P�@     ��@     P�@     ��@     Ё@     H�@     �@     Px@     Pz@     px@     0v@     �r@     �r@      o@      o@      k@     �m@     �h@     �f@     `e@     �a@     �`@     @^@     �Z@     @Y@      X@      [@     �R@     @R@     @W@     @R@     @Q@      J@      K@     �K@     �M@      H@      D@     �G@      :@      @@      8@      8@      6@      ;@      =@      3@      7@      3@      *@      $@      "@      4@      (@      5@      1@       @      (@      (@      $@      @       @      &@      @       @      @       @      @      @       @      @      &@      &@       @      @       @      @       @      @      @      �?       @       @      @       @      @      �?      @      @      @      @      �?      �?      @      �?      @               @       @       @      @      @              �?              �?      �?      �?       @       @       @      �?              @      �?              @      �?      �?      �?              @      @      �?              �?      �?              @              �?              �?      �?              �?       @              �?              �?              �?              �?               @      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?      @      @              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?               @      �?              �?      �?      �?       @       @              �?              �?      �?      �?              @              �?      @      �?       @              �?               @      �?      �?      �?      @               @              �?       @              �?      @              @              @               @      @       @      @               @              @      @      �?      @       @      �?      �?      @      @      @       @      �?       @       @      @      @       @      @      �?      @      @      @      @      @       @       @      @      @      @      "@      @      @       @      $@       @      $@      @      @      &@      &@      $@      &@      "@      $@      @      "@      0@      *@      *@      6@      3@      .@      5@      2@      4@      :@      1@      4@     �@@      C@      ?@     �D@      D@      F@      M@      K@     @P@     �I@      O@     �P@      S@     �R@     �U@     @W@     @Y@     �Y@      \@      \@     �`@      c@     �d@     �f@      k@     �i@      i@     �p@     �o@     �s@     u@     �u@      z@     0|@     p}@     ��@      �@     ��@     @�@     (�@     ��@     0�@     x�@     H�@     X�@     ��@     ܔ@     @�@     |�@     l�@     H�@     ,�@     ��@     ��@     �@      �@     ��@     ج@     D�@     X�@     �@     \�@     ��@     �@     �@     ?�@    �Q�@    ���@    ���@    �7�@    ���@     ��@    �O�@    ���@    ��@    @�@     :�@    �F�@    ���@    �J�@    ��@    �Q�@    ���@    ���@     ��@    ��@     ��@     ��@    @p�@     ��@     C�@    @��@     ��@    ��@     ��@     �@     r�@    ���@    ��@     )�@    ���@    ���@    �G�@    �-�@    ��@     ��@    ���@     ��@     ��@     �@    �H�@     ��@    ���@    ���@    ���@     <�@     ��@     �@     ��@     "�@     ��@     ǵ@     ��@     b�@     l�@     4�@      |@     Pp@     @^@     �Q@      D@      8@      &@      ,@      "@      �?      �?        
�
predictions*�	    G�׿    �=@     ί@!  <5q
;�)N�����D@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�x?�x��>h�'�����%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�iD*L��>E��a�W�>pz�w�7�>I��P=�>>h�'�?x?�x�?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?cI���?�P�1���?3?��|�?�E̟���?ܔ�.�u�?��tM@�DK��@{2�.��@�������:�              �?              �?       @      �?      �?       @      @              @      .@      2@      6@     �@@     �F@      H@     �B@     �N@      N@      W@     �X@     �[@     �Z@     @`@     ``@     �]@     �`@     �`@     ``@     �]@     �X@     @V@     �Y@     @T@     @W@     �Q@      K@      M@      F@      D@      E@      B@      <@      ;@      8@      ;@      >@      (@      .@      9@      "@      *@      0@      ,@      @      "@      @      &@      @       @      "@       @      @       @       @       @      @      �?      @      �?               @      �?              �?              @      �?      �?       @       @              �?              �?              �?              �?              �?              �?              �?              �?               @      �?              �?      �?      �?      �?              �?       @              �?      @       @      @       @       @      �?      �?      @      �?      �?      @      �?      @      @      @      @      "@      @       @      @      "@      2@      $@      *@       @      (@      3@      *@      ,@      5@      5@      0@      9@      6@      7@      5@      A@      >@      A@      ?@      G@     �A@      @@      ?@      J@      F@      @@      B@     �C@      A@      C@      3@      ;@      @@      B@      8@      9@      6@      7@      6@      .@      *@      ,@      *@      (@      "@       @      "@       @      @      @      @      @      �?      @       @       @      @      @      @      �?              �?              �?              �?              �?        �yX�b.      X
�	�j+x���A#*�\

mean squared errorĔ=

	r-squared���>
�C
states*�C	   @2�    ��@    ��=A!����&���)`8�N;1�@2�!�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c�nx6�X� ��f׽r����tO����f;H�\Q������%���/�4��ݟ��uy�i@4[���Qu�R"�PæҭUݽH����ڽ�
6������Bb�!澽5%����G�L�������/���EDPq��|_�@V5����M�eӧ���@�桽�>�i�E������z5��!���)_�����_����e���]����x����\��$��%�f*��-���q�        �-���q=!���)_�=����z5�=��s�=������=�>�i�E�=��@��=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��.4N�=;3����=���X>�=H�����=PæҭU�=�Qu�R"�=�/�4��==��]���=��1���=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�!              �?      @      @      "@      4@     �@@     �E@     @Y@      o@     P~@     `�@     ԡ@     ��@     ��@     ��@     ��@    �r�@     ��@    �9�@    ���@    ���@     �@     ��@     ��@    �%�@    ���@    ��@     h�@     )�@     �@    ���@    ��@     �@    ���@    ��@     X�@     F�@     ��@    �&�@    �n�@     ��@     ��@     ��@    �~�@    @-�@    @��@     2�@     c�@     �@    @<�@    �<�@    @q�@    �2�@    �B�@    �B�@    �-�@    @4�@    @��@    @��@    @��@    @!�@    �c�@    �I�@     �@    �Z�@     w�@    ���@    ���@     ��@    �{�@     ޿@     �@     ��@     ϸ@     Ӷ@     Ҵ@     C�@      �@     �@     ��@     (�@     v�@     �@     ��@     |�@     j�@     H�@     ��@     �@     ؖ@     ��@     ��@     ��@     �@     ��@      �@     ��@     ��@     Ї@     �@      �@     8�@     @     �|@     �y@     �x@     0u@     �t@     �t@     �p@     �n@     �l@     `i@      h@     �e@     �d@     `d@     ``@     `a@     �`@     �\@     �W@     @W@     @V@      P@     @W@      S@     �Q@     �I@      L@      L@     �E@      B@     �K@     �J@      B@      B@      9@     �D@     �@@      ;@      0@      7@      .@      8@      .@      6@      9@      .@      @      7@      $@      *@      (@      "@      $@      0@      @      ,@      &@      @      @      @      &@      @      $@      @      @      @      @      @      @      @      @      @      @      @      �?      @      @      @      @       @      @      @      @      @      �?       @      �?      @       @      �?      �?      �?       @       @       @              �?       @       @      @      @               @      �?      �?      @      �?       @              @      @      @      �?      @              @       @      �?              @      �?              @      @      @      �?              @               @       @              �?       @      �?      �?               @              �?      @      �?              �?      �?      �?              �?              �?               @              �?              �?      �?              �?              @      @              �?              �?              �?              �?              �?               @              �?      �?              �?              �?              �?              �?               @      �?              �?       @              �?       @       @              �?      �?              �?               @      �?      �?              @              �?      @              �?              �?      �?       @       @      �?      @      �?              �?      �?      �?       @      �?              �?       @      �?       @       @               @       @       @       @       @      @      �?      �?      @      @      @       @       @       @      @      @      @      @       @       @      �?      @       @      @      @      @      @      @      @      @      @      @       @      @      @       @      "@      @      "@      (@      @      "@      $@      2@      (@      ,@      $@      $@      .@      3@      3@      4@      6@      <@      3@      :@      <@      :@      9@      A@      @@      C@     �E@     �F@      M@     �E@      N@     �J@     @P@     �O@      N@     �S@     �V@      U@     �W@     @V@     �^@      \@     �d@     �b@     `c@      e@     �e@     `f@     `k@     �l@     `p@     �q@     pr@     @u@     �w@     �v@     �|@     �@     (�@     ��@     �@      �@      �@     @�@     `�@     ��@     l�@     @�@     ��@     ��@     ��@     ��@     ��@     ��@     l�@     ��@     ��@     B�@     :�@     Ϊ@     0�@     ʰ@     n�@     �@     #�@     a�@     ݹ@     	�@     �@    ���@     (�@     p�@    �P�@    �9�@    ���@     �@     ��@     �@    ���@     ��@    �y�@    ���@     .�@    @�@    �2�@    �}�@     7�@    @��@     |�@    ���@    �4�@    @�@    ���@    ���@     ��@    �I�@    ���@     r�@    @�@    �(�@     �@     F�@     ��@    �q�@     
�@     ��@    �j�@    ���@     2�@     ,�@     ��@     ��@    �%�@     -�@    �p�@    ���@     ��@     ��@    �z�@    ���@     �@     4�@     ��@     $�@     [�@     ��@     :�@     ��@     ��@     y@     �n@      a@      T@     �F@      1@      2@      *@      @      @      �?        
�
predictions*�	   ��}ҿ   @��@     ί@!  ���^A@)�� \��F@2�_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�ji6�9���.���vV�R9��T7����FF�G �>�?�s������m!#���
�%W��E��a�W�>�ѩ�-�>��(���>a�Ϭ(�>>h�'�?x?�x�?�5�i}1?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?�DK��@{2�.��@�������:�              �?              �?      @       @      @       @       @      ,@      &@      .@     �A@     �F@     �H@      H@      T@     �R@      R@     @W@      `@      Y@     �Z@     @[@     �W@     @[@      X@     @V@      S@     �P@     @S@     �G@     �I@      G@     �K@      B@      D@      ?@      =@      :@      =@      :@      7@      ,@      4@      ,@      *@      0@      &@      "@      0@      @      @      $@      @      "@       @      @       @       @      �?       @       @      �?      @      @       @      �?       @       @      �?              �?      �?      �?      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?       @              �?       @       @       @      �?      �?              �?      �?              �?      @              @      @      �?      @      @      @      @      @      @      @      @      "@      @      "@      $@      @      *@      (@      .@      "@      .@      0@      .@      6@      >@      3@      6@      <@      >@      A@      A@     �@@     �A@     �K@      N@      J@      K@     �E@     �I@      M@     �I@      O@     �P@      N@      O@     �L@      N@      D@      H@      F@      E@     �E@      D@      @@     �F@      >@      ?@      6@      0@      .@      2@      (@      (@       @      @      @      @      @      @      @      @              @              �?       @       @              �?              �?              �?        ]�S".      YgB	�^x���A$*�\

mean squared errorH�=

	r-squared$��>
�C
states*�C	   ` ?�   ��@    ��=A!eN;��-��)��'F��@2�!h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[��PæҭUݽH����ڽ���X>ؽ��
"
ֽ(�+y�6ҽ;3���н�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%�����Į#�������/���EDPq��<QGEԬ�|_�@V5����M�eӧ�_�H�}��������嚽��s�����:���-���q�        �-���q=��@��=V���Ұ�=��M�eӧ=|_�@V5�=����/�=�Į#��=��؜��=�d7����=�!p/�^�=��.4N�=��
"
�=���X>�=H�����=PæҭU�=i@4[��=z�����=�/�4��==��]���=��1���=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�!               @      @      ,@      >@      ?@      P@     �a@     `o@     8�@     X�@     Т@     ��@     /�@     ��@    �k�@     ��@    �8�@    ���@     �@     ��@    �_�@    ���@    �4�@     n�@     ��@     ��@    ���@     ��@    �U�@    ���@    �y�@     "�@     ��@     ��@    �T�@    �z�@     ��@    �:�@     O�@    ���@    �k�@     ��@    ���@    ���@    �-�@    �l�@    �_�@    �S�@    ��@    ���@    ���@    ���@    ���@    �"�@    �&�@    @k�@    ���@     6�@    @�@     
�@    ���@     ��@     	�@     ��@     ��@     ��@    �c�@     ��@    �]�@     �@     g�@     ��@     շ@     �@      �@     �@     ��@     ή@     ��@     ȩ@     ��@     "�@     @�@     ^�@     X�@     ��@     ��@     ��@     ,�@     �@     ��@      �@     ��@     ��@     P�@     x�@     H�@     @�@     ��@     ��@     ��@     �{@     @|@     �x@     pu@     �r@      s@     `o@     �p@      j@     �l@     �h@     �f@     �d@     @c@     @_@     `a@     �`@     �\@      X@     �W@      X@     �S@     @T@     @T@     �N@      G@      E@     �L@      J@      B@      G@     �D@      A@      <@      ;@      3@      <@      7@      6@      2@      3@      4@      :@      $@      .@      7@      0@      .@      *@      .@      "@      ,@       @      &@      (@      (@      @      @      "@      @      @      @      (@      &@       @       @      (@      @      @      @      @      @      @      @       @      @      @      @      @      @      @      @       @      @      @      @      @      @      �?              �?      @      @       @       @      @      �?              @              �?      @      �?              @       @      @      �?              �?              @      @      �?      �?      �?               @              �?      �?              �?       @      �?              �?      �?              �?      �?      @              @              �?      �?      �?               @              �?              �?      �?      �?              �?              �?      �?      �?      �?      �?              �?              �?      �?              �?      �?              �?      �?      �?              @      @              �?              �?              �?              �?      @      �?              �?               @              �?              �?      �?              �?      �?      �?               @      �?       @              �?              �?       @      �?      �?      @              �?               @      �?       @      �?      �?      �?      �?      �?       @      @              @              �?               @              �?       @      @              �?      @              @       @       @               @      �?       @              �?              @       @       @       @      @      @      @      @      @      �?      @       @       @      @      @       @      @       @       @       @      @      @      @      @      @      @      $@      @      @      &@      @      *@      *@      "@      @      "@      0@      .@      @      "@      3@      1@      (@      >@      4@      "@      5@      9@      6@      7@     �@@      8@      7@      ;@      =@      ?@     �B@      E@     �F@      Q@     �M@     @P@      P@     @R@     �P@     �Q@     �T@     @Z@     �Y@     @Z@     @\@      a@     �b@     @d@      e@      e@     �g@     @m@     p@     �p@     �q@     0s@      u@     �w@      y@     �z@     �|@     �~@      �@     ��@     @�@     ��@     ��@     ؉@     (�@     Џ@     0�@     \�@     Е@     D�@     ��@     T�@     ��@     4�@     ֡@     �@     ��@     ��@     �@     �@     �@     �@     G�@     ��@     ٶ@     ��@     ʻ@     ��@     u�@    ���@     ��@    �s�@    �x�@    ���@     ��@     �@     ��@     x�@    �]�@    �"�@    �#�@    ��@    �)�@     l�@    @h�@    �w�@     y�@    @0�@     |�@    @!�@     e�@    @^�@     U�@    ��@     ��@     V�@     ��@     �@     �@     ��@    @]�@     ��@     ��@    �t�@     ��@     ^�@     �@     ��@    �3�@    �c�@     ��@    ���@    �;�@    �|�@      �@     Y�@     ��@    �[�@     R�@     �@     	�@     Ϲ@     [�@     (�@     ��@     ֭@     ��@     ��@     �z@     @m@     �a@     �U@     �H@      5@      2@      (@      @      @      �?        
�
predictions*�	   @�Hۿ   ��_@     ί@!  p\Q�)"ʘ�J@2�W�i�bۿ�^��h�ؿ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
������E��a�Wܾ�iD*L�پ�_�T�l׾+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?{2�.��@!��v�@�������:�              �?               @       @      �?      @      @      @      ,@      $@      5@      <@     �B@      D@     �@@     �I@     @P@      I@      S@     �Z@     �\@     �`@      [@     @[@     `a@     �`@      ^@     �^@     @Z@     �X@     �S@     @R@     @P@      G@     �K@      K@     �G@      I@      I@      @@      6@     �A@      4@      <@      9@      1@      3@      &@      0@      4@      4@      "@      @      $@      "@      "@      @      @      @      $@      �?      �?      @      @      @      �?      @      �?      @      @      @      �?      �?              �?              �?      �?              �?              �?       @      �?      �?      �?      �?              �?      �?              �?      �?               @      �?              �?      @       @      @      �?      @      �?      �?      @      @       @      �?      @      @      @       @      @      @       @      @      @      $@      "@      @      "@      $@      .@      "@      .@      0@      *@      2@      :@      7@      6@      7@     �A@      :@      <@      C@      ?@      C@     �E@      8@     �B@      ?@     �B@     �@@      I@      H@      I@      =@     �G@      E@     �@@      <@     �H@      B@     �@@     �@@      3@      6@      4@      5@      0@      ,@      2@      ,@      3@      @      ,@      &@       @       @      ,@      &@      @       @      @      @       @              @      �?       @       @              �?      �?      �?              �?              �?        �v��2.      E/	cC�x���A%*�\

mean squared error�=

	r-squared>��>
�C
states*�C	   ���   @�L@    ��=A!0Pi#��) 	�. �@2�!�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r���f;H�\Q������%��=��]����/�4��z�����i@4[����
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�
6������Bb�!澽G�L������6���Į#����EDPq���8�4L���|_�@V5����M�eӧ���@�桽�>�i�E�������嚽��s����x�����1�ͥ��\��$��%�f*�z����Ys��-���q�        �-���q=�>�i�E�=��@��=V���Ұ�=y�訥=�8�4L��=�EDPq�=����/�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�d7����=�!p/�^�=��.4N�=�|86	�=��
"
�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��='j��p�=��-��J�=�K���=�9�e��=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>�i
�k>%���>��-�z�!>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�!              @      *@      1@      B@     �N@     @a@     @m@     �}@     ��@     ��@     @�@     �@     �@    ���@     ��@    �&�@     ��@     |�@     |�@    �|�@    ���@    ���@     ��@     ��@     /�@     ��@    ���@    ���@     �@     ��@    �B�@    ���@     ;�@    �9�@    �4�@     ��@    ���@    �s�@    @m�@    �B�@    @��@    ���@    ���@     �@    @��@    �&�@    ���@    ���@    ���@    @_�@    @��@    �o�@     M�@    @�@    ���@    @��@    �x�@    ���@    @��@     ��@     �@    ���@    ���@     ��@    ���@    �~�@     ��@     ��@     ��@     ��@     T�@     3�@     ��@     ѳ@     C�@     İ@     ��@     Ȭ@     ©@     f�@     �@     ��@     p�@     �@     ��@     T�@     ��@     И@     ��@     T�@     ��@     ��@     8�@     �@     (�@     ��@     x�@     ��@     ��@     �@      }@     Pz@     �y@     `t@     @s@     pv@     �p@     �m@     �n@     �n@      h@     �g@     @e@     �e@     @b@     �_@      a@     �V@     @V@     �Y@      U@     �R@     �V@     �P@     �N@      K@      M@      L@     �J@     �H@      A@      >@     �L@      =@      E@     �A@      >@      9@      6@      4@      3@      6@      3@      4@      *@      *@      *@      1@      $@      ,@      6@      &@      $@      $@       @      (@      *@      @      $@      @      @      "@      @      "@      @      @      @      @       @      @       @      @      @      �?      @       @       @      @              @      @      @      �?       @      @      @       @      �?      �?       @      @              @      �?      @      @              @      @       @      @       @      @       @      �?      @       @      @      @       @      @      �?      @       @      �?      �?       @      @               @      @      �?      �?       @              @              �?       @      �?              �?      �?               @      �?              �?              �?              �?              �?      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      @      �?              �?              �?               @      �?              �?      �?              �?               @       @               @               @       @      �?               @      �?              �?      �?      �?              �?      �?      �?       @              �?               @      �?      �?              �?               @       @              �?      �?      �?              �?               @      �?       @      @      @      �?              �?      @      �?              �?      �?      �?      �?               @      �?      @      @      �?      @      �?               @      �?       @      �?       @       @       @      �?      @      @      @      @       @       @      @      "@      @      @      @       @      "@      @      @      @      @       @      @      $@      "@      @      @      .@      2@      @      @      ,@      $@      $@      (@      ,@      0@      &@      5@      .@      8@      3@      9@      :@      ;@      @@      7@      <@      6@     �A@     �@@      E@      :@     �H@     �K@     �I@     �M@     �K@      N@     �P@     �S@     �V@     �S@     �Y@      \@     �]@     �a@     �a@      a@     @e@     �c@      e@     �g@     @k@     �k@      p@     �q@     �s@     �v@      w@      x@     �{@     �{@     H�@     X�@     P�@     �@     ��@     H�@     H�@     X�@     ��@     �@      �@     x�@     ��@     ̘@     X�@     ��@     �@     ȡ@     P�@     ~�@     d�@     @�@     ��@     Ư@     p�@     ڲ@     ��@     �@     Ӹ@     Ѻ@     ��@    ���@     ��@     ��@     ��@    ���@    ���@     X�@    �g�@    ���@    ���@    ���@    @�@    ���@    @Z�@    ���@    @��@    @��@     ��@     �@     ��@    @m�@    ���@    @��@     ��@    ���@    ���@    @�@    �o�@     "�@     $�@     /�@    �p�@    ���@     +�@     u�@     ��@    ���@    ��@    ���@     ��@     ��@     ��@    � �@    ���@     ��@     ��@     ��@     ��@     ~�@     ��@     �@     ݾ@     O�@     ��@     �@     �@     ��@     ��@     %�@     ��@     �~@     �q@     �e@      Y@     �O@     �@@      &@      &@      (@       @      �?        
�
predictions*�	   ���տ   �
�@     ί@!  �����)����}zL@2���7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9����ڋ��vV�R9���d�r�x?�x��>�?�s���O�ʗ�����|�~���MZ��K����Zr[v�>O�ʗ��>x?�x�?��d�r?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?�iZ�?+�;$�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@{2�.��@!��v�@�������:�               @              �?              �?       @      @      @      *@      $@      3@      6@      :@      C@      C@      H@     �L@     @S@     �R@      W@      W@      [@     �V@     �_@     �X@     �Z@     �W@     @Z@     �Y@      W@      W@      T@     �T@      P@      P@      T@      N@     �K@      D@     �G@     �E@     �C@      <@      ?@      9@      2@      2@      ,@      5@      0@      *@      2@      "@      @       @       @      "@      @      (@      @       @      @      @      @       @      @       @      @      @      @       @       @       @              �?      @      �?              �?              �?              �?              �?              �?              �?              �?               @              @              �?               @       @      @       @      @      �?       @              @       @      @      @      @      @      @      @      @      �?      @      "@      @      (@      4@      *@      $@      &@       @      :@      ,@      5@      7@      1@      4@      <@      >@      ;@     �A@     �E@     �D@      G@      I@     �L@      F@      I@     �C@     �A@     �G@      E@      C@      B@     �C@      E@      A@      >@      9@      C@      <@      9@      7@      2@      0@      6@      0@      0@      *@      &@      *@      "@      @      (@      @      @      @      @      @      @      @       @              �?      �?      @              �?               @       @              �?      �?              �?        ���.      E$B�	&�x���A&*�\

mean squared error/�=

	r-squared:L�>
�B
states*�B	   �ڗ�    �@    ��=A!V�W*č��)��4o��@2� h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��`���nx6�X� ��f׽r���f;H�\Q������%���9�e����/�4��ݟ��uy�z������|86	Խ(�+y�6ҽ;3���н��؜�ƽ�b1�ĽK?�\��½�
6��������6���Į#�������/���EDPq��|_�@V5����M�eӧ���@�桽�>�i�E��_�H�}��������嚽��s���-���q�        �-���q=�8ŜU|=%�f*=̴�L���=G-ֺ�І=!���)_�=����z5�=��s�=������=_�H�}��=�Į#��=���6�=G�L��=5%���=�Bb�!�=K?�\���=�b1��=��.4N�=;3����=��
"
�=���X>�=H�����=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��=��1���='j��p�=�K���=�9�e��=����%�=f;H�\Q�=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>%���>��-�z�!>4�e|�Z#>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�                @      $@      &@      8@     �C@     �O@      Y@     �l@     �~@     �@     ��@     G�@     �@     ��@     e�@    �?�@     ��@    ���@    �^�@     ��@     ��@     C�@    �J�@     7�@    ��@     �@     ��@    �D�@    ���@    �V�@    �u�@     B�@     ��@     q�@     i�@    ���@     '�@    ��@    @�@    @|�@     ;�@    ���@    �T�@    ���@     ��@    ��@    ��@    @�@     ��@    @n�@    �I�@    @g�@    ���@     h�@    @o�@     t�@    ���@    �=�@    ���@    �J�@    @T�@    ���@    ���@    ���@     X�@    ���@    �2�@    �m�@     ��@    �^�@     ν@     ��@     Ҹ@     �@     ��@     ��@     m�@     ¯@     �@     �@     Χ@     ��@     �@     R�@      @     h�@     �@     �@      �@     ��@     �@     X�@     P�@     ��@     �@     p�@     (�@     (�@     ��@     �@     @@     `�@     �y@     py@      w@      t@     �t@     Pr@     �o@     �n@     �k@     `h@     �b@     `g@     @c@     �`@     �`@     �_@     @\@     �Y@     �T@      X@      R@     �U@      U@      J@     @P@     �N@      I@      L@      I@      B@     �D@     �C@      @@      7@      >@     �@@      3@      ?@      8@      5@      =@      3@      4@      &@      *@      1@      5@      .@      *@      .@      $@      &@      *@      "@      "@      @      @      ,@      @      (@      "@      @      @      @      @      @      @      @      @      $@      @       @      @      &@      @       @      @       @      @       @       @       @      @       @       @      @       @       @      @       @      @       @       @      @      @       @      @      @              �?      @       @      �?       @       @              �?      @              �?      �?               @       @      �?      @      �?       @      �?              �?      �?      �?              �?       @       @      �?              �?      �?              �?      �?              @      @              �?       @              �?               @              �?              �?              �?              �?              �?      �?              @      @              �?              �?               @              �?      �?               @              �?      �?               @               @              �?       @               @      �?      @       @              �?              �?              �?              �?      �?              �?              �?              @              �?              �?              @      �?              �?       @       @              �?      �?      @      �?              �?       @               @      @               @      �?      �?       @       @      @       @      �?              @       @       @      @       @      @      @       @              �?       @      @       @      @      @      �?      @      @      @      @       @      @       @      (@      @      @      @      @      @      @      (@      "@      @      @      *@      (@      "@      .@      @      5@      ,@      @      1@      2@      8@      *@      2@      ,@      8@      5@      :@      7@      1@      @@      A@      <@      A@     �B@      H@      G@     �L@     �L@     �K@     @Q@      M@     �P@     @R@     �R@     @T@     �W@     @Z@     @Z@      ]@      a@      a@     �b@      d@     �d@     �f@     �i@      l@      j@      q@     Pp@     �t@     pu@     `v@     �y@     �|@     0}@     ��@     ��@     H�@     �@     ��@     ��@     ��@     (�@     D�@     ̒@     ,�@     �@     ̗@     |�@     ��@     �@      @     ,�@     ޣ@     (�@     $�@     d�@     ��@     q�@     �@     p�@     6�@     ��@     ��@     	�@     ܾ@     �@    �7�@     P�@     V�@    �5�@     0�@     �@    ���@    ���@    �k�@     ��@    @F�@     ��@    @?�@    ���@    ���@    ���@    @��@    @A�@    @�@    @��@     :�@    �R�@     b�@    @L�@    ���@    ���@     �@    ���@     ��@    ��@    @��@    ���@    ���@     t�@     �@    ���@     O�@     ��@     s�@     ��@    ���@     y�@     ��@    �5�@    �P�@     ��@    �B�@    ���@    ���@     ��@    �$�@     W�@     ޼@     q�@     L�@     ��@     Y�@     ��@     l�@     8�@     �u@     �j@      ^@     �X@     �B@      *@      .@      (@      @      �?        
�
predictions*�	   ��oӿ   ��
@     ί@!  b��,@@)�
F:��D@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'���FF�G �>�?�s����h���`�8K�ߝ뾄iD*L�پ�_�T�l׾�uE����>�f����>>�?�s��>�FF�G ?x?�x�?��d�r?��ڋ?�.�?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�Š)U	@u�rʭ�@�������:�               @      �?      �?              @       @      @      *@      *@      *@      ;@      6@      <@     �C@      ?@     �E@     �I@     �K@     �H@     �N@     �S@     �S@     �V@     �W@     �U@     �Q@      U@      U@     �S@     @R@      R@      O@     �R@     �O@     �P@     �E@      M@      C@     �C@     �B@      @@      <@      7@      0@      4@      (@      .@      0@      0@      *@      @       @      $@      "@      @      &@      @       @      @      @      @       @      @       @       @       @       @      @      @      �?               @       @      @      �?      �?              �?              �?      �?              �?      �?      �?              �?              �?              �?              �?              �?              �?              �?               @              �?       @       @              @       @      �?      �?      �?      @      �?      @      @      @      @       @      @      �?      @      @      @      *@      $@      (@      @      &@       @      2@      *@      "@      *@      3@      4@      5@      6@      4@      B@      >@     �H@      F@      J@      D@     �I@     �H@     �J@     �M@     �J@     �L@     �P@      L@      N@     �J@      N@      O@      P@      D@     �B@      K@     �J@     �G@      G@      F@      D@     �E@      9@      ;@      :@      3@      2@      5@      3@      (@      $@      0@      ,@      @      @      @       @       @      @      @       @      @      @      �?       @       @      �?      �?       @       @              �?        ��#��.      GP��	���x���A'*�]

mean squared error�r=

	r-squared���>
�C
states*�C	    �U�   �0�@    ��=A!��տ�&��)w����@2�!h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���f��p�Łt�=	��J��#���j�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K���'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L������:������z5�����_����e���]����x����z����Ys��-���q�        �-���q=�/k��ڂ=̴�L���=�1�ͥ�=��x���=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=|_�@V5�=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=K?�\���=�b1��=�d7����=�!p/�^�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�!               @      @      *@      3@     �H@      J@     �T@     `n@     �}@     ؐ@     f�@     ��@     ѷ@     ��@     ��@     ��@    ���@    ���@     ��@    ���@     ,�@    �N�@    ���@    ��@     ;�@     ��@    ���@    ���@     @�@     ��@    ���@     ��@     )�@    ���@     ��@    ��@     ��@     E�@     @�@    �4�@    @��@    �W�@    ���@    �&�@    ���@    ���@    �'�@    �	�@    ���@     ��@    �y�@    �	�@    ��@    �9�@    @��@    ���@     ��@    �Q�@    ���@     ��@     p�@    �O�@    � �@     &�@    ���@     ��@     ,�@     ��@    ���@     ��@     M�@     ��@     A�@     W�@     �@     �@     �@     R�@     0�@     :�@     0�@     ��@     J�@     ��@     ��@      �@     P�@     ��@     X�@     ��@     |�@     �@     �@      �@     ��@     Ї@     X�@     ��@     ��@     H�@     �}@     �{@     �y@     pw@     �s@     �s@     �s@     �r@     �m@     `j@     `l@     @f@      h@      c@     �c@     @b@     ``@     �Z@     @X@     �T@     �T@      R@      Q@      U@     �R@     �Q@     �I@      I@     �O@     �M@      >@     �E@      C@      @@      >@      :@      A@      :@      8@      8@      4@      >@      ,@      *@       @      *@      (@      (@      8@      &@      1@      .@      (@      $@      *@      "@      *@      1@      @      @       @      @      @      @      $@      @      @      @      @      @      @      @      @      @      @       @       @      @      �?      �?       @      @       @      �?       @       @      @              @      @               @      @      @      @      @      �?      �?      @       @              @       @              @       @              �?      @      �?       @      @       @               @       @      @      �?              �?               @               @              �?      �?               @              @      �?       @              �?               @              �?              @              �?      �?       @       @              �?      �?       @      @              �?              �?              �?              �?      �?              �?      @      @              �?              �?              �?      �?       @      �?      �?               @              �?      �?              �?      �?              �?              �?               @      �?              �?      �?              �?       @              �?      @       @       @      �?       @       @      �?              �?              �?      �?      �?       @       @               @      �?      �?      @              @       @      �?      @      @      �?      �?               @              @      @               @              @      �?      �?              @      �?      �?      @       @      @      �?               @      @      @      @      �?      @      �?               @      @       @       @      @      �?      @      @      @      @      "@       @      @      &@      $@      @      @      @      @      @       @      @      "@      .@       @      @       @      ,@      0@      $@      0@      *@      $@      *@      (@      0@      5@      0@      ,@      8@      5@      :@      :@      C@      8@      ;@      6@      ?@      ?@      I@      @@     �B@      D@     �B@      D@     �M@     @Q@     �Q@     @T@     @S@     @V@     @T@      Y@      [@     �Z@     �`@      _@     �c@     �c@     �d@     `f@      h@     �j@     �l@     Pq@     �q@     �s@     �u@     u@     �v@      y@     �z@     h�@     H�@     ��@     8�@     ��@     ��@     ��@     ��@     �@     �@     P�@     P�@     ԕ@     ��@     T�@     ��@     ��@     Ң@     D�@     �@     p�@     V�@     >�@     F�@     ̰@     !�@     }�@     
�@     ޸@     ��@     ��@     V�@     �@    ���@     q�@    ���@     �@    �W�@     5�@     :�@    ���@    @}�@    �N�@     "�@     ��@     �@    @�@    ��@    ���@    ���@    �l�@    �_�@    @_�@    �J�@    @��@     ��@    ���@     T�@    @S�@    �K�@    @��@     ��@    �Q�@    �n�@     ��@    ��@    �1�@    ���@     +�@    ��@    ���@    ���@    ��@    ���@     �@    ���@     �@     �@     ��@    �#�@    ���@    �|�@     .�@     ��@     `�@     ��@     ʸ@     H�@     ��@     ��@     ��@     �@     �x@     �m@      c@     �V@      I@      3@      ,@      "@      @      �?        
�
predictions*�	   ��ӿ   �W�@     ί@!  <ߙ1@@)Cl�v��H@2��Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�+A�F�&�U�4@@�$��.����ڋ��vV�R9��T7���x?�x��>h�'��>�?�s���O�ʗ����ߊ4F��>})�l a�>>h�'�?x?�x�?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?yL�����?S�Fi��?��tM@w`<f@�DK��@{2�.��@�������:�              �?      �?       @              @       @      @       @      1@      ,@      1@      6@      ?@      B@      G@      M@      J@     �S@      T@     @R@     @U@      [@     �W@     �Y@     �Y@     �V@     �V@     @W@      T@     �Q@     �W@     �O@     �Q@      M@     �E@      J@      J@     �G@      A@     �@@      B@      5@      9@      8@      ;@      1@      ,@      ,@      1@      @      4@      $@      $@      (@      $@      @      @      @      �?      $@      @      @      @      @      @              @      @      @       @      �?      �?      @              �?      �?              �?               @       @      �?               @              �?              �?              �?              �?              �?      �?               @       @      �?               @              �?      �?       @      �?      �?      @       @      @      @      @      @      @      @       @      �?       @      @      @      @      $@       @      �?      &@       @      &@      @      ,@      0@      2@      1@      6@      4@      :@      1@      2@      =@      :@      B@      <@      C@      E@     �G@      A@      F@      G@     �G@      J@     �D@     �I@     �I@      H@     �H@      D@      G@      J@     �M@      G@     �F@     �G@      ?@      @@     �B@      6@      B@      @@      >@      6@      1@      1@      *@      *@      &@      (@      @      $@      "@      @      @      @      �?       @       @               @               @              �?              �?              �?              �?        ���4�/      \t�	ӥ	y���A(*�^

mean squared error�1=

	r-squared̞�>
�E
states*�E	   �*�    D�@    ��=A!�o������)!�6��@2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	�2!K�R���J����"�RT��+��y�+pm��mm7&c��`���f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽G�L������6���Į#����EDPq���8�4L���V���Ұ����@�桽��s�����:������z5��!���)_�����_�����1�ͥ��G-ֺ�І��-���q�        �-���q=z����Ys=%�f*=\��$�=�1�ͥ�=��x���=e���]�=���_���=����z5�=���:�=�>�i�E�=��@��=V���Ұ�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=��1���='j��p�=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>2!K�R�>��R���>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              �?      @      (@      3@     �A@     �O@      X@      h@     |@     ��@     �@     "�@     	�@     ޸@     �@    ���@     �@    �}�@     ��@     ��@     G�@    ���@    �b�@     ��@     ��@     /�@    ���@    ��@     ��@     s�@     )�@    �`�@    �O�@     ��@    �f�@    ���@    ��@    �w�@    �"�@     ��@    @��@    @C�@    �7�@    ���@    ���@     �@     ��@     ��@    ���@     ��@    @p�@     U�@    @Q�@    �T�@    @��@     ��@    ���@     S�@    ���@    �1�@    @�@    �y�@     =�@     ��@     ��@    �l�@    �8�@    ���@     [�@     -�@     ��@     ]�@     I�@     9�@     ��@     β@     ϰ@     ��@     ��@     ��@     ԧ@     \�@     R�@     N�@     �@     ��@     ؛@     ��@     P�@     @�@     ��@     4�@     t�@     �@     �@     ��@     ��@      �@     @�@     h�@     �@     @{@     y@     0{@      x@      t@     �q@     pr@     `p@     �k@      j@      i@      g@     �d@     �c@     �b@      a@      ]@      ]@     @^@      U@      W@     �U@     �T@      Q@      P@     �P@     �O@      M@     �H@     �G@      @@     �E@     �B@     �C@     �A@     �@@      ?@      A@      8@      =@      3@      9@      1@      2@      0@      3@      "@      *@      2@      *@      ,@      &@      @      $@       @      "@      0@      "@       @       @      �?      @       @      @      @      @      @      @       @      @      @      @      @      @      @       @      @       @      @      @      @      @       @      @      @      �?      @       @      �?      @       @      @      @       @      @      @       @      @      @      �?      @              @       @              @              �?              �?              @              �?              �?      �?      �?              �?              �?       @              @               @      �?              @               @              �?       @              �?       @              �?       @              �?       @              �?       @      �?      �?               @               @      �?              �?               @       @              �?              �?              �?      �?               @              �?              @      &@      �?              �?              �?       @      �?              �?              �?       @              �?              @      �?      �?              �?      �?      �?      �?              �?              �?      �?      �?      �?      �?              �?              �?              �?      �?      @      @       @      �?      �?      @      �?       @               @              �?       @              �?              �?              �?       @      @              �?       @       @      @      @       @              �?      @      �?      @      �?      @      @       @      @              �?      @      �?      @       @       @       @      @      @      @       @      @      @      @       @      @      �?              @      @      @      @      @      @      @      @      @      @       @      *@      @      *@      &@      &@      "@      @      @      .@      @      (@      &@      &@      1@      2@      1@      0@      0@      3@      ,@      4@      5@      7@      ;@      ,@      >@      8@      7@      7@      ;@      =@      >@      E@      C@      @@      L@      H@     �O@      E@     �M@     �M@     �L@     �S@     @Q@     @Y@     �X@      Z@     �^@     @[@     �c@      ^@      d@     @b@     `e@     @g@     �i@     �n@     @o@     �p@     �s@     r@     t@     �w@     �x@     �z@     �|@     0�@     ��@     @�@     ��@     ��@     h�@     ��@     h�@     ��@     ��@     �@     ��@     \�@     l�@     ��@     `�@     ��@     ^�@     (�@     x�@     @�@     f�@     l�@     �@     r�@     ɳ@     ��@     $�@     ι@     ��@     ��@    ��@    ���@     d�@     ��@    �Y�@     p�@     ��@    ��@    ���@    ���@    ���@    @��@    @g�@    ��@    �R�@    �"�@    @��@    ��@     #�@    ��@    ���@    ���@     ��@    @��@     l�@    @��@    �6�@    ���@     ��@    ���@     ��@    ���@     U�@     �@     ��@    �j�@    ���@     ��@    ���@    ���@     ��@    ���@    �(�@      �@     Q�@     u�@    �]�@     S�@     ��@    �|�@    �(�@     ��@    �u�@     о@     �@     w�@     �@     β@     ��@     ؘ@     0�@     Pv@     �n@      _@     �Q@     �E@      4@      "@      @      @        
�
predictions*�	   ��Zٿ   `�@     ί@!  ���@)��⻫_H@2�W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���ڋ��vV�R9������6�]���>�?�s���O�ʗ���>�?�s��>�FF�G ?��[�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�Š)U	@u�rʭ�@�DK��@�������:�              �?      �?               @              @      @      @      (@      (@      3@      &@      (@     �@@     �A@      E@      I@      C@      M@      N@      V@     �T@     �Y@     �Y@     �Y@      [@     �X@     �Y@     �Y@     �W@      U@     @S@      X@     @T@     �R@     �Q@     �M@     �G@      E@     �J@      ?@      B@     �D@      6@      8@      <@      1@      1@      "@      &@      3@      $@      2@      .@      $@      @      "@      @      @      @       @      @       @       @       @      @      @      @      @      �?      �?      @      �?      �?               @       @              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?              �?      �?      �?      �?      �?      �?      @       @      @              @              �?      �?       @              �?       @      @               @      @       @      @      @      @      @      @      (@      @      &@      @      @      2@      1@      @      4@      $@      4@      4@      .@      8@      C@      8@      >@     �@@      A@      A@      E@      ?@     �D@      F@     �E@      E@     �E@      H@      K@      H@      N@      J@     �G@      A@      G@      M@     �F@     �@@      A@     �C@      9@     �@@      :@      5@      9@      .@      4@      ,@      .@      &@      @      "@      @      @      @      @      @      @      @      @              �?      �?      �?       @      �?               @               @              �?      �?        ���-B/      ^��=	д1y���A)*�^

mean squared error'�=

	r-squaredfƚ>
�C
states*�C	   ����   �ð@    ��=A!�C�����)��7�w�@2�!�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�4�e|�Z#���-�z�!��i
�k���f��p�Łt�=	��J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO�����9�e����K��󽉊-��J�'j��p�ݟ��uy�z�����i@4[���Qu�R"���
"
ֽ�|86	Խ(�+y�6ҽ;3���нK?�\��½�
6������Bb�!澽5%�������6���Į#�������/����M�eӧ�y�訥�!���)_�����_����e���]��G-ֺ�І�̴�L�����8ŜU|�x�_��y��-���q�        �-���q=�8ŜU|=%�f*=\��$�=�/k��ڂ=�1�ͥ�=��x���=e���]�=���_���=!���)_�=�>�i�E�=��@��=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�
6����=K?�\���=�b1��=��؜��=�!p/�^�=��.4N�=(�+y�6�=�|86	�=���X>�=H�����=PæҭU�=i@4[��=z�����=ݟ��uy�==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�!              @      0@      4@      A@      L@     @W@     `l@     @{@      �@     ��@     Y�@     :�@     ȸ@     �@     9�@     ;�@    ���@    �4�@     ��@    ��@     7�@    ���@    ��@     9�@     i�@    ���@    ���@     ��@     I�@     N�@     }�@    �n�@    �j�@     ��@    ���@    �,�@     ��@    �w�@    @�@    ���@    ���@    @��@    �o�@    ���@    @��@    @8�@    ���@    @��@    @��@     ��@    �d�@    �X�@    ���@    ���@    ���@    �e�@    @��@    �'�@     h�@    ���@     �@     q�@    ���@    �=�@     e�@    �g�@    �'�@     2�@     �@     1�@     �@     ��@     R�@     ��@     ۱@     ��@     &�@     ª@     �@     ��@     �@     ��@     ��@     f�@     �@     ܙ@     ��@     Ȗ@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@      �@     8�@     ��@      ~@     �}@     |@     �w@     �u@     �u@     @s@     �q@     pp@     �k@     �g@     �e@      g@      f@      a@     @^@     �]@     �^@     �Z@     @Y@      Y@     �S@     �U@     �Q@     �O@      Q@     �P@     �H@     �C@     �I@     �H@     �C@     �J@      C@     �@@      ;@      ;@      =@      9@      :@      ?@      5@      4@      .@      7@      $@      0@      ,@      ,@      *@      $@      ,@      &@      @      @      (@      $@      (@      @      @      @      @      @      @      $@       @      @      @      @       @      @      @      @      &@      @      @      @      @      @      @       @      @      @       @      @      @      @      @       @      @      @      �?       @       @              @       @      �?      �?       @       @      �?      �?       @      �?      �?       @      �?       @       @       @      �?       @       @              �?              @       @              �?      �?       @       @       @      �?      @      �?              �?              �?      �?      �?               @              �?              �?              �?              �?      �?      �?               @      �?              �?              �?      �?              �?              �?              @      @              �?      �?      �?              �?              �?      �?              @              �?              �?              �?              @      �?       @              @              �?              @      �?              @       @              @               @      �?      �?      @              �?      �?      @              �?      �?              @               @      @              �?               @      �?       @      �?       @      �?              �?      �?      �?              @       @      @      @      @      @              @      @      @       @       @              @      @      �?      @      @       @      @       @      @               @      �?      @       @      @      @       @      @      @      @      @      �?      @      "@       @      @      @      "@      @      @      @      @      @      @      $@      @       @      "@      @      @      @      &@       @      (@      &@      ,@       @      @      2@      &@      *@      (@      3@      3@      2@      8@      ;@      >@      1@      ?@      =@      :@      4@      C@      9@      A@      C@      G@      C@      H@     �M@     �O@     �G@     @S@     @R@     �Q@     �U@      T@      V@     �Z@     �]@      Z@      `@     `b@     �c@     �b@      g@     �f@      l@      m@     �n@     �q@     `q@     pt@     �r@     �v@     �{@     p}@     p{@     `@     �@     x�@     H�@      �@     p�@     ��@     `�@     (�@     ��@     ��@      �@     t�@     ܙ@     d�@     H�@     Ġ@     ��@     ��@     j�@     ܨ@     L�@     Ĭ@     A�@     v�@     ��@     }�@     ��@     ��@     ��@     ݾ@    ���@    �t�@     �@    ��@     ��@     ��@    ���@    �h�@    @�@    @4�@    @8�@    @��@    ���@    @��@    �*�@    ���@     ��@    @��@    �{�@    ���@    @��@    ���@    �a�@     N�@    @H�@    ���@    @��@     ��@     8�@    ��@    ���@     ��@    ���@    @��@    �c�@    ���@     ��@    ��@    ���@     ��@    �:�@     z�@    �t�@    ���@     ��@     3�@     I�@     ��@     r�@     ��@     X�@    ���@     ��@     �@     ؽ@     {�@     �@     ��@     ��@     ��@     �@     Ps@      g@     �Y@      M@      <@      2@      @      "@      @        
�
predictions*�	   `�-�   �N9@     ί@!  ���"#�)�PYI@2�+Se*8�\l�9⿰1%࿗�7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��.����ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
��FF�G �>�?�s���I��P=��pz�w�7��['�?��>K+�E���>6�]��?����?�vV�R9?��ڋ?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?u�rʭ�@�DK��@�������:�              �?      �?              �?       @       @      @      @      @      *@      @      .@      5@      8@      =@     �G@     �E@      K@      P@      R@     @S@     @U@     �X@     @X@     �Y@     �Z@     �]@      Y@     @^@     �]@     �[@      W@     @T@      U@     �Q@      M@     �J@      M@      A@     �M@     �A@      C@      ?@     �A@      5@      6@      1@      7@      3@      ,@      0@      (@      "@      &@      @      @      @      @      @      "@       @      @      @              @       @      @              @       @      �?      @       @      @      �?      @               @      @      �?               @       @              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?       @      �?      �?               @              �?       @              @       @       @      �?      @       @      �?       @      @      @      @      @      @      $@      @       @      "@      @      &@      @      @      &@      (@      0@      1@      3@      1@      2@      0@      7@      9@      7@     �D@      @@     �C@      7@      F@     �A@      B@      E@      F@      F@     �G@      E@     �F@      G@     �F@      F@      A@     �J@      E@     �C@      >@     �B@      ;@      :@     �A@      :@      @@      ;@      2@      0@      4@      .@      $@      .@      &@      "@      @      @       @      @      �?       @      @      @      @      @      @      �?      �?      �?      �?              �?      �?       @      �?              �?        D�I�/      @lj�	]�]y���A**�_

mean squared error�L=

	r-squared8��>
�F
states*�F	   �8`�    ��@    ��=A!�8��#S��)H�7���@2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r�����-��J�'j��p���1���=��]����/�4��ݟ��uy��|86	Խ(�+y�6ҽ;3���н��.4Nν�
6������Bb�!澽5%����G�L������6���Į#�������/���EDPq���8�4L���<QGEԬ�|_�@V5����M�eӧ���@�桽�>�i�E��_�H�}��������嚽��s�����:������z5��!���)_�����_����e���]���1�ͥ��G-ֺ�І��/k��ڂ�\��$��%�f*��8ŜU|�x�_��y��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=\��$�=�/k��ڂ=e���]�=���_���=!���)_�=����z5�=���:�=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=����/�=�Į#��=���6�=G�L��=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              @      @      &@      *@      6@     �J@     @[@      k@      y@     ،@     h�@     )�@     �@     R�@      �@     \�@    �b�@    ���@    ��@    ���@    �/�@     ?�@    ���@    �v�@    ��@     �@     ��@    ���@    ���@     ��@     ��@    ��@     (�@    �@�@    ���@     ��@     .�@     6�@    ��@    ���@    ���@    ���@    ���@    �A�@    �E�@    @��@     ��@    ���@     b�@    @O�@     ��@    ��@    �P�@    @<�@    @6�@    ���@     K�@     ��@     ��@     e�@    ���@     ��@     ��@     ��@     �@     ��@     	�@     i�@     x�@     ��@     ۹@     ��@     �@     ]�@     ��@     L�@     ��@     �@     V�@     �@     j�@     ��@     ��@     x�@     �@     �@     4�@     �@     ��@     H�@     �@     ��@     (�@      �@     ��@     ��@     Ȃ@     �@     ��@     �}@     �|@     �z@     Pw@     �u@     @u@      r@     `o@     �m@     �p@     �i@     �g@     �g@     `e@     �d@     �]@      a@     @`@      ^@     �X@     �V@     @U@     �S@     @Q@      S@     �M@     �O@      R@     �J@      E@     �G@      @@      H@     �C@      B@      <@     �@@      @@      7@      7@      ;@      =@     �@@      1@      2@      9@      .@      3@      4@       @      *@       @      0@      0@      0@      .@       @      $@      "@      @      @      &@      @      @      @      @      @      @      @       @      @      @      @      @      "@      @      @      @      @      @       @      @      @      @      @       @      @       @      @      @      @       @              �?       @      �?      @              �?      @      �?       @      @      �?      @              @       @      @       @              �?              @              �?      �?      @              @               @      @      �?               @       @      �?              �?              �?              �?      �?      �?              @      �?       @              �?               @              �?               @              �?       @      �?              �?      �?       @               @              �?      �?      �?              �?      �?      �?      �?      �?              �?              �?      �?               @              *@      @              �?      �?              �?               @       @       @       @              �?      �?              �?       @               @      �?       @              �?              @              �?       @              �?      �?      �?      @              �?               @              @               @              �?              �?       @              �?               @      @      �?      �?       @      �?               @      �?       @       @      �?      �?       @               @      �?       @      @      �?      @      @              �?      �?      @      �?       @      @      @              @      @      @      @      @      @      �?       @       @      @      @      @      @      @      @      �?      @      @      @      �?              @      @      @      @      @      @       @      @      @      (@      "@      @      @      &@       @      "@      @      @       @      @      "@      .@      $@      $@      5@      $@      $@      9@      3@      .@      *@      5@      6@      4@      7@      :@      ;@      >@      <@     �C@     �A@     �@@     �@@      @@     �F@     �G@      I@      I@     �J@     �H@      K@     �O@      N@     @S@     �Q@      W@     �X@      V@     �[@     �Y@     ``@      `@     �a@     �c@     �c@     �g@      i@     `k@     �m@     �o@      r@     �q@     �u@     t@     �z@     y@      {@     `}@      @     P�@     Ȅ@     �@      �@     ��@     Ȋ@     ��@     p�@     ��@     ��@     <�@     ��@     �@     \�@     P�@     ��@     ��@     ��@     �@     V�@     ��@     ��@     l�@     �@     �@     ?�@     ��@     ȹ@     �@     .�@     .�@     5�@    ���@    ���@    ���@     ��@     ��@    �Z�@     	�@    @%�@    �`�@    @l�@     ��@     x�@    �)�@    �L�@     ��@    @/�@     >�@    ���@    ���@    ���@     W�@    @%�@     +�@    ���@    ���@    ���@     Q�@    ���@    ��@    @�@     �@     ��@    �,�@     c�@    ���@    �z�@     �@    ���@    ���@     +�@    �|�@     ��@    ��@    ���@     ��@     ��@     /�@     p�@    ��@     ��@    ��@     ��@     �@      �@     }�@     K�@      �@     8�@     `q@     @b@     @U@      M@     �@@      *@      @       @       @        
�
predictions*�	   `1}�   �x�@     ί@!  �,o)�)]�s+I@2�uo�p�+Se*8俗�7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"���ڋ��vV�R9��T7����5�i}1�1��a˲���[���FF�G ��ߊ4F��h���`iD*L�پ�_�T�l׾U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�E̟���?yL�����?S�Fi��?ܔ�.�u�?u�rʭ�@�DK��@�������:�              �?              �?       @      �?       @               @       @      @       @      ,@      9@      ?@     �D@     �G@      F@      K@      S@     �W@      X@     ``@     �_@     �b@      a@     �`@      `@     @b@     �a@     �`@     @X@      V@     �Q@      S@     �N@     �L@     �J@     �H@     �C@      B@      B@     �@@      >@      9@      1@      9@      ,@      4@      "@      .@      "@      3@      (@      @      @      @      *@       @      @      @      $@      @      "@       @      @       @      @      @       @      @       @      �?      @      @              �?              �?               @              �?      �?      �?              �?      �?              �?              �?               @      �?      �?              �?      @      �?      �?      @      @      �?      �?      �?      @       @      �?      @      �?      �?       @      @      @      @      @      @      @      (@      @      @       @      .@      "@      .@      (@      (@      1@      6@      4@      3@      1@      1@      3@      2@      4@      :@      =@      @@      <@     �@@     �@@      C@      :@      =@      B@      >@      ?@     �D@      8@      A@      B@      :@     �B@      3@     �E@      ?@      ;@     �D@      <@      9@      2@      7@      ,@      ,@      (@      0@      $@      &@      $@      (@      @      "@      @       @      @       @       @      �?      @      @      �?      �?      �?              �?      �?      �?              �?        q3�0      xiB	Lm�y���A+*�_

mean squared error��=

	r-squared���>
�E
states*�E	   @���   @@    ��=A!cB�A���)?����'�@2�"�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#�%�����i
�k���R����2!K�R��Z�TA[�����"�RT��+���mm7&c��`���nx6�X� ��f׽r�������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��ݟ��uy�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽K?�\��½�
6�����5%����G�L���y�訥�V���Ұ����@�桽�>�i�E��e���]����x�����1�ͥ��G-ֺ�І�x�_��y�'1˅Jjw��-���q�        �-���q=z����Ys=x�_��y=�8ŜU|=%�f*=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=!���)_�=����z5�=���:�=��s�=��@��=V���Ұ�=y�訥=��M�eӧ=�8�4L��=�EDPq�=�Į#��=���6�=G�L��=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=;3����=(�+y�6�=���X>�=H�����=PæҭU�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              @       @      1@      ?@     �D@     @Z@     �n@     z@     ��@     ��@     �@     K�@     ��@     J�@    �G�@     y�@    ��@    ���@    �l�@     M�@     ��@     ��@    ���@     ��@    ��@    �8�@     �@     &�@     ��@     -�@      �@     P�@     e�@    ��@     �@    ��@     �@     S�@     #�@    @0�@    ���@    @��@    @r�@    �z�@    @W�@    ���@    ��@     �@     ��@     ��@     ��@    ���@    ���@    @��@    ���@    ���@    @��@    ���@    @��@     ��@    ���@    ���@     I�@    ���@    �:�@    ���@     ��@    �b�@     Կ@     �@     ��@     ·@     =�@     �@     ��@     �@     Z�@     ��@     �@     p�@     6�@     ֣@     >�@     J�@     �@     Ț@     �@     ؖ@     `�@     ԓ@     ��@     ��@     P�@     `�@     ��@     �@     H�@      �@     ؀@     p�@     �}@      |@     �w@     �w@     �s@     pt@     Pq@     `o@      l@      i@     �j@     �e@     `e@      d@     �b@     �Z@     �^@      ]@      \@     �V@      W@      U@     �U@      O@     @P@      J@     �Q@     �N@      G@      H@      F@     �E@      =@     �A@      A@      A@      :@      ;@      <@      5@      4@      8@      6@      2@      3@      :@      6@      *@       @      (@      *@      ,@      &@      (@      ,@      "@      (@      @      .@      @      "@      $@      $@      "@      @      @      @      @      @       @      @       @      @      $@      @      @      @      @      @       @      @      @      @      @       @      @      @      @       @      @      @      �?       @      �?      @      @      �?      @      @      �?      @              �?       @       @              @      �?      �?              �?       @      �?       @              �?      @      �?      �?              �?              �?              �?      @              �?               @              �?      �?              �?      @      �?      �?      �?              �?      �?      �?              �?       @              �?              �?              �?              �?              �?              �?              �?              �?              $@      @      �?               @      �?              �?               @              �?              �?              �?              �?              �?              �?      �?               @      �?      �?              �?      �?              �?              �?      �?              @       @      �?              �?              �?      �?      �?              �?      �?      �?      �?       @      @       @      �?       @      @      �?      �?      @      @       @       @               @      @      �?       @      @      �?       @       @       @      �?      @      �?      @      �?       @      �?       @      @       @              �?              �?      �?      @       @      @      @      �?      @      @              @      @      @      @      @      @       @      @       @       @      @      @      @      @      @      @      @      @       @      @       @      $@      @      &@      $@      @       @      @      "@      &@      ,@      (@      0@      $@      *@      0@      0@      6@      ,@      .@      3@      &@      1@      7@      7@      ;@      ;@      9@      2@     �B@      ?@      B@     �B@      B@      G@     �D@      D@     �E@      K@      M@      Q@     �Q@     �M@     �R@     �T@     @X@     �V@     @\@      Y@     �^@     �[@      `@      c@     @e@      h@     �i@      h@     `l@     �l@      n@     �r@     �t@     �r@     �v@     �y@     @y@      ~@     �|@     �@     ��@     h�@     ��@     �@     �@     x�@     ��@     ��@     Ē@     @�@     l�@     И@     ��@     h�@     L�@     �@     &�@     ��@     
�@     *�@     F�@     �@     ̰@     I�@     ų@     ��@     r�@     S�@     ��@     ӿ@     ��@     �@     �@    ���@    �_�@     �@    ���@    ���@    @z�@     J�@     i�@     C�@    ��@     ��@    @��@    @��@     ��@     ��@    ���@    �	�@    ��@     	�@    @#�@    @��@    �Q�@    ���@    @J�@     s�@     ��@    ��@    ���@     ��@    �1�@    @>�@     Q�@    ���@    ���@     ��@     �@     ��@    ���@     �@    ���@    ��@    �Q�@    ���@     ��@    �P�@    �F�@    ��@     ��@     ��@     ��@     ��@     �@     Z�@     K�@     ��@     w�@     X�@     �@     p@      d@     @U@      F@      <@      0@      $@      @      @        
�
predictions*�	    seؿ   ���@     ί@!  p��A@)���K�I@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��>�?�s���O�ʗ���>�?�s��>�FF�G ?��d�r?�5�i}1?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?S�Fi��?ܔ�.�u�?w`<f@�6v��@u�rʭ�@�DK��@�������:�              �?      �?      �?      @      �?      �?      �?      @      @      $@      "@      @      (@      0@      8@      6@      F@      K@     �C@      L@      L@      R@     �V@     @T@     @U@     �T@     �V@      W@     @U@      T@     @U@     �W@     @Q@      V@     �N@     �S@      O@      F@      L@     �D@     �H@     �B@      A@      @@      9@      9@      7@      3@      3@      $@      5@      .@      (@      (@      2@      $@       @      @       @      @      @       @      @       @      @      @       @      @      @      �?               @      @      �?       @      �?      �?               @              �?              �?              �?              �?              �?      �?              �?              �?              �?               @              �?      @      �?               @      �?              �?       @              @      �?      @              @       @      @       @      @       @      �?      �?      @      @      "@      @      @      @      @      @      @       @      .@      @      0@      ,@      1@      4@      3@      0@      6@      :@      7@      =@      =@     �@@      E@      ;@      B@      A@     �D@     �G@      D@      K@     �K@      J@     �O@      O@     �K@      K@      L@     �K@      H@     �D@     �N@      K@      B@     �E@     �E@     �D@     �@@     �@@      6@      5@      :@      3@      2@      ,@      ,@      (@      $@      ,@      @      @      @      @      $@      @      @      @      �?      �?      �?      �?      �?               @              �?       @              �?              �?              �?        ��0      xiB	��y���A,*�_

mean squared error-�=

	r-squared���>
�E
states*�E	   ����    ��@    ��=A!^�r����)|�7�D�@2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[���tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L����Į#�������/���EDPq��<QGEԬ�|_�@V5�����_����e���]��̴�L�����/k��ڂ�\��$��%�f*�x�_��y�'1˅Jjw��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�/k��ڂ=̴�L���=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=�Į#��=���6�=G�L��=5%���=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              @       @      "@      2@      B@      J@      Z@     �l@     �z@     p�@     ��@     ѹ@     �@     ��@    �z�@     ��@     p�@    ���@    ���@    ��@     ?�@     `�@     ��@     B�@     ��@     L�@    �[�@     �@    ��@     ��@    �-�@     6�@    �+�@    ��@     v�@     ��@    �t�@     ��@    �F�@    ���@    @a�@    ���@     ��@    ���@    @G�@    �L�@    ���@    ��@     &�@    @��@    @��@    �{�@     l�@    �N�@     w�@     ��@    @m�@    �n�@     �@    @X�@    �;�@     ��@     ��@    �u�@    ���@     �@     V�@      �@     W�@     ��@     �@     %�@     շ@     I�@     ��@     9�@     �@     8�@     ��@     ֪@     4�@     b�@     ��@     ��@     Z�@     ̞@     ��@     x�@     ��@     ��@     �@     ��@     p�@     p�@     X�@     ��@     �@      �@     �@     ��@     �@     @�@     @{@     Px@     �v@     �t@     `s@      s@     �o@     �n@      l@      i@     `h@      h@     @e@     @a@     �_@     @]@     �Z@     @]@     �]@      X@     �T@     �O@     @U@     @P@     @R@     @P@     �L@     �J@     �E@     �D@      D@      =@      B@      C@      ?@     �A@      =@      ?@      7@      ?@      1@      =@      7@      *@      *@      5@      5@      4@      .@      ,@      1@      $@       @      *@       @      @      .@       @      "@      (@       @      @      @       @      @      @      (@      @      @      @      @      @       @      @       @      @      @      @      @      @      @      @      �?      �?      @      @      @       @      @       @      @      @              @      @      �?      @       @       @       @      �?      �?      �?      @      @      @      @       @      @       @               @      �?              �?       @       @      @      �?      �?       @       @      �?      �?      �?      �?      �?      �?               @               @      �?      �?       @      �?      �?      �?              �?      �?      �?      �?              �?      �?              �?              �?       @              �?      �?      �?              �?      �?              �?               @               @              �?              �?              $@      3@              �?      �?      �?               @              �?              �?      �?               @              �?       @      �?      �?              �?              �?               @              �?      �?              �?              �?      �?              �?       @       @       @       @              �?      �?       @      �?              �?      @       @      �?       @      �?              �?      @       @               @      �?      @              �?       @       @      �?      @      �?      @       @      @      �?      @      �?      �?      �?      �?       @              �?      @      @      @       @      @      �?       @      @      @      @      @      @      @      @      @      �?      @      @       @      @      @      @      @      @      "@      @       @      @      $@      @      "@      $@       @      @      "@      *@      @      "@      .@      $@      6@      ,@      1@      @      $@      *@      4@      6@      3@      6@      ,@      :@      9@      3@      <@      ?@      >@      =@      A@     �B@      7@      A@      E@      A@     �G@      J@      M@      C@     �J@      O@     �G@      S@     @Y@     �S@     �U@     �V@     �]@      _@     @]@     �\@     @`@     �b@     �d@     @h@      h@     `i@     �n@     0p@      n@     �p@     �q@     �s@     pv@     �w@     �}@     0y@     P@     �@     ��@     ��@     ��@      �@     ��@     ��@     X�@     ̑@     @�@     �@     D�@     Ę@     ��@     t�@     F�@     ��@     ��@     �@     ��@     �@     J�@     ��@     ��@     ]�@     �@     0�@     H�@     ƺ@     ��@     l�@     z�@    ���@     ��@     �@    �<�@     ��@     ��@     s�@    @@�@    @<�@     ,�@    �,�@    @j�@    �G�@    ���@    @c�@     ��@    ���@    ���@     ��@    �=�@    ��@    @��@    ��@    @_�@    ���@     c�@     ��@    @!�@    @k�@     ��@     -�@    �k�@     y�@     ��@    ���@    ��@    �W�@    ���@     2�@     ��@    ��@    ���@     i�@     ��@     ��@     ��@    ���@    �3�@    ��@    �X�@     ��@     l�@     ��@     ��@     .�@     �@     �@     Z�@     ؖ@     ؁@     `q@     `c@      W@      G@      A@      "@      .@      @      @        
�
predictions*�	   ��|�   �0S@     ί@!  b1! C@)��L21M@2�+Se*8�\l�9��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&��[^:��"��S�F !�1��a˲���[���h���`�8K�ߝ뾮��%ᾙѩ�-߾;�"�q�>['�?��>���%�>�uE����>��Zr[v�>O�ʗ��>>�?�s��>����?f�ʜ�7
?>h�'�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?��tM@w`<f@u�rʭ�@�DK��@�������:�              �?              �?      �?      �?       @              �?      "@      (@       @      1@      6@      .@      6@      B@      F@      E@     �K@      O@      M@     @W@     �S@     �T@     �T@     �U@      U@     �S@      S@     �Q@      V@     �R@      S@     �N@     �M@     �I@     �M@      J@     �D@     �G@      E@      E@      :@      5@      4@      6@      6@      <@      1@      0@      *@      $@      ,@      .@      (@      @      @      @      @      @      @       @       @      @      @      @               @      �?      @       @       @       @      @              �?       @      �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?               @               @      �?      �?               @              �?      @      �?       @               @              @      @      @              @      @      @      @      @      @      @      "@      @      $@      "@      @      @      0@      (@      1@      1@      3@      1@      4@      7@      ,@      <@      8@     �@@      A@      C@      <@      B@     �B@      J@     �J@      F@      L@      P@      L@     @Q@     �M@      L@     @Q@      O@      N@      J@     �M@     �E@      B@      F@     �F@     �H@      F@      @@      ;@      B@      ;@      9@      <@      8@      6@      $@      0@      @      $@       @      @      @      $@      @      @      @      @      @       @      @      �?       @               @       @      �?              �?      �?      �?              �?        ��3�/      _�ɯ	��y���A-*�]

mean squared errorsw=

	r-squared�Y�>
�C
states*�C	   �/�   ��E@    ��=A!�ѠdO���)�Ow��3�@2�!h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r�������%���9�e�����-��J�'j��p���1���=��]����/�4��ݟ��uy�i@4[���Qu�R"ཱ�.4Nν�!p/�^˽����/���EDPq���8�4L�����@�桽�>�i�E�����:������z5��!���)_�����_����e���]����x�����1�ͥ��G-ֺ�І�'1˅Jjw�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=:[D[Iu='1˅Jjw=x�_��y=�8ŜU|=%�f*=���:�=��s�=y�訥=��M�eӧ=�Į#��=���6�=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�!              �?       @      @      2@     �A@     �I@      _@     �j@     pz@     ��@     0�@     ��@     d�@     ��@    ���@     	�@     ��@     ��@    ��@    ���@     ��@    �3�@     ��@     ��@    ���@    ���@    ���@    ���@     I�@    ���@    ���@    �%�@    ���@     I�@    �}�@    �>�@     ��@     ��@     ��@    �a�@     ��@    @u�@    �3�@    @��@    @��@    ���@    ���@    ���@    @�@    ���@     ��@    �H�@    @��@    ��@    @��@     ��@    �E�@    �-�@    �O�@     �@    ���@    ���@    �z�@     ��@     ��@     @�@    �m�@    ���@    �L�@     v�@     ��@     ̸@     ��@     �@     /�@     \�@     �@     �@     �@     H�@     ��@     ��@     �@     x�@     ̟@     ��@     8�@     ��@     d�@     ��@     4�@      �@     (�@     ��@     ��@     ��@     Ѕ@     Є@     (�@     @�@     p~@     `}@     0x@      x@     Pv@     �t@     @s@     r@      p@     �p@      l@      g@     `h@     �e@     �d@     @b@     �b@     �\@      _@     �Y@      V@      W@      T@     @P@     �T@     �P@      O@     �R@     �F@      F@     �D@     �D@     �B@      D@      D@      A@      C@      ;@      <@      4@      3@      3@      0@      7@      0@      2@       @      (@      1@      *@      2@      (@      5@      @       @      &@      0@      @      "@       @      @       @      @      @      @       @      $@      �?      "@      @      &@       @      @              @      @      @      @      @      @       @      @      @      @      @               @       @       @      �?       @      @      �?      @              �?      �?       @      @      @       @      �?      @       @       @      �?              �?      @      �?       @      �?      �?      @      @      @       @       @              �?      �?              �?      �?       @      �?              �?      �?       @      �?      �?               @              �?      �?      �?      �?       @              �?              @              �?      �?               @              �?      �?              �?      �?              �?              �?      �?      �?      $@      @              �?      �?      �?      �?      �?              �?              �?               @              �?       @      �?              �?              �?      �?              �?       @              �?      �?      �?      @               @      @              @       @              @              �?              �?      @       @       @       @      �?       @      @       @      �?       @              �?      �?       @       @              @      �?       @       @       @      @      �?      @      �?      @      �?      @      @      @      �?      �?      �?      @      @      @      @       @      �?      @       @      @      @      @      @      @      @      @      @      @      @      �?      @       @      @      @      @      @      $@      @      "@      (@       @      @      "@      $@      &@      .@      @      @      ,@      .@      "@       @      (@      0@      (@      $@      4@      1@      .@      5@      7@      2@      5@      9@      2@      B@      =@      4@     �E@      @@     �D@      B@      F@     �D@     �B@     �C@      J@      J@      J@     �O@      N@     �K@     �R@     �S@     @U@     @R@     @W@     �\@     @Z@     �`@     �^@     �[@      `@     �d@      f@      g@      h@     �j@     �n@     �n@     �m@     0r@     �r@     0u@     �v@     pz@     @y@     �~@      @     Ȁ@     (�@     �@     p�@     p�@     ��@     �@     $�@     ��@     \�@     t�@     p�@     �@     ��@     �@     4�@     r�@     z�@     �@     R�@     `�@     V�@     �@     �@     7�@     ٴ@     H�@     ��@     ��@     V�@     �@    �F�@    ��@    �	�@     F�@     `�@    ��@     0�@    �G�@    �4�@    �c�@     �@    ��@    ���@    ���@    �8�@    �,�@    �;�@    �r�@    �r�@    @��@    ���@    � �@    @+�@    �'�@    ��@    ���@    @U�@    @��@     L�@    �=�@     ��@    @��@    ��@    @c�@     9�@    �z�@     ��@    ���@     ��@    ���@     ��@     J�@    �l�@    �^�@     ��@     ��@    ��@     <�@    �	�@    ���@     ��@    �I�@    �3�@     ��@     ��@     G�@     ;�@     d�@     0�@     (�@     �o@     �b@      T@      K@      ?@      7@      @       @      �?        
�
predictions*�	   `$3�    ?�@     ί@!  �u�@)nYY��P@2��1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+��S�F !�ji6�9���ѩ�-�>���%�>�f����>��(���>})�l a�>pz�w�7�>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?3?��|�?�E̟���?yL�����?S�Fi��?��tM@w`<f@زv�5f@��h:np@�������:�              �?              �?              �?              �?      @      @      @      @      4@      .@      6@      =@     �D@      E@     �J@     �D@     @P@     �J@     @V@     �S@     �X@     �V@     @V@     �X@      X@     �S@     �Z@     �V@     �Y@     @V@     �U@      Q@      N@     �P@     �I@      N@     �J@      F@      B@      H@     �A@      <@      >@     �@@      9@      1@      6@      (@      (@      *@      0@       @      &@      $@       @      @      @       @       @      @      @      @      @      @      @      @      @      @      �?              @               @      �?               @       @               @               @              �?              �?              �?              �?              �?      �?      �?              @      �?              �?              @               @              �?       @       @      @      �?      @      �?      �?      �?      �?      @      @       @      @      @      @      @      @      @      @      @      (@      ,@      ,@      @      &@      &@      2@      0@      .@      .@      6@       @      ,@      :@      5@      A@      @@      >@     �D@     �F@      E@     �L@     �G@     �G@     �E@      G@     �B@      J@      L@     �F@      J@     �B@      D@     �G@     �F@     �@@      >@      :@     �F@      >@     �A@      =@      8@      7@      8@      *@      4@      0@      2@      *@      (@      "@      "@      �?      @      @      @      @              @      @      @      �?       @              �?              �?      �?      �?              �?              �?        �)�?�0      {��&	�iz���A.*�a

mean squared error�=

	r-squared���>
�F
states*�F	   �ء�   ��@    ��=A!�^ۥ_��)�咵!�@2�#�6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!��i
�k���f��p�Łt�=	���R�����J��#���j�Z�TA[��RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K���'j��p���1���=��]���ݟ��uy�z�����i@4[��H����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�b1�ĽK?�\��½���6���Į#����EDPq���8�4L���|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E��_�H�}��������嚽��s�����:���/k��ڂ�\��$���-���q�        �-���q=z����Ys=:[D[Iu=x�_��y=�8ŜU|=%�f*=\��$�=�/k��ڂ=̴�L���=G-ֺ�І=�1�ͥ�=��x���=���_���=!���)_�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=��M�eӧ=�8�4L��=�EDPq�=���6�=G�L��=5%���=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>y�+pm>RT��+�>���">Z�TA[�>�#���j>��R���>Łt�=	>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�#              @      ,@      .@     �A@     �K@      ]@     �i@     �z@      �@     ��@     ˹@     ��@     ��@    ���@    �e�@     ��@     >�@    ���@    ���@    ���@    ���@     ��@    ��@     ��@    ���@     7�@    ���@     (�@    ���@     ��@     7�@    ���@    �G�@    �
�@    � �@    ���@     ��@     ��@    ��@    ���@    ���@    �q�@    ���@    �1�@     ��@     ��@    @��@    �n�@    ���@    ���@     ��@    ���@     ��@    ���@    @��@     Y�@     E�@    ���@    �v�@     ��@    ���@     k�@    ���@     �@    ���@    ���@    ���@     6�@     d�@     G�@     ��@     �@     D�@     R�@     �@     �@     ��@     ĩ@     ��@     D�@     أ@      �@     *�@     �@     t�@     �@     �@     �@     ��@     ��@     �@     ��@     8�@     x�@     `�@      �@     ��@     X�@     �@     {@     �z@     �w@     �w@     `t@     �u@     �r@     `p@     �k@     `j@      j@     @f@     �d@      d@      b@     �a@      _@     @_@     @[@     �Z@     �U@     @U@     �W@     �N@     �R@     �P@      K@     �N@      N@     @P@     �H@      F@      H@      A@      E@      @@      =@      @@     �B@      :@      ?@      9@      6@      :@      3@      ,@      *@      1@      *@      *@      @      .@      .@      *@      "@      ,@      @      (@      *@      @      "@      @      @      .@      &@      "@      ,@      @      @      @      @      @      @      "@      @      @      @      "@      @      @      @      @      @      @      @      @      �?               @       @      @      �?      �?      @      @      �?      �?      �?      @      @               @              �?       @       @              �?       @       @              �?      @      �?              �?              @      �?               @              �?      �?      �?              �?      �?              @      @               @      �?              �?      @      �?      �?              �?      �?               @      �?              �?              �?              �?      �?              �?               @              �?              �?       @              �?       @      �?       @              �?              �?              (@      $@              �?              �?      �?       @       @      �?      �?              �?              �?               @      �?       @      �?       @              �?              �?              �?      �?              �?               @      �?      �?      �?      �?      �?       @              �?       @       @       @       @      �?              �?              �?               @      �?       @       @      @       @      �?      @              �?       @      �?      �?               @      �?      �?      �?      �?       @               @      @      @      @       @               @       @      @      @      @      @       @      @      @               @      @       @      @      �?       @              @      @      @      @      @      @      @      @      @      @       @      @      @      @      @      @      "@      @       @       @       @      @      @      @      "@      @       @      "@      @      ,@      @       @      @       @      @      0@      1@      ,@      $@      *@      .@      *@      (@      3@      4@      4@      .@      ,@      1@      9@      <@      >@      ;@      4@      D@      1@      F@     �A@      A@      >@      D@      E@      C@      L@      I@     �M@     @P@     �P@      P@      P@      T@      V@     @T@     �U@     �X@     �X@     �]@     @_@     �`@     �b@     �d@     �b@     @e@     �g@     @f@     �i@     �l@      q@     `q@     Pt@     0t@     �r@     �v@      z@     �{@     �@     x�@     ��@     x�@     �@     ��@     ��@     @�@     h�@     �@     ��@     �@     P�@     ��@     t�@     ��@     ��@     ��@     f�@     ܤ@     
�@     H�@     ��@     ��@     k�@     �@     t�@     �@     ķ@     �@     ��@    ��@     `�@     ��@    �,�@     �@    ��@    ���@     ��@    @ �@    ���@    @.�@     :�@    �	�@    ���@    �n�@    @�@    ���@    ���@     ��@    ���@    @��@    � �@    @X�@    ���@    @��@     V�@    @��@    �+�@    �k�@    ���@    ���@    @��@    ��@     �@     6�@    �%�@    ���@    ��@     A�@    ���@    ���@     ?�@    ���@    ���@     ��@    ���@     ��@    ���@    �i�@    ���@     '�@     ��@    �G�@    �|�@     ��@     (�@     ��@     ��@     �@     ̕@     �@     �n@     �b@      Q@      M@      >@      @      @      @      @        
�
predictions*�	    �e�   �ų@     ί@!  ��l�$�)��� �+P@2�uo�p�+Se*8�W�i�bۿ�^��h�ؿ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�6�]���1��a˲�8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>����?f�ʜ�7
?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@!��v�@زv�5f@�������:�              �?              �?              @      �?      @       @      @      *@      6@      4@      <@     �F@      G@      I@     �Q@      N@      Q@     �W@     @V@     �Y@     �W@      [@      X@     @[@     �[@     �a@     @Z@     �\@     �W@     �X@     @R@     �Q@     @Q@     @Q@     �M@     �Q@     �B@      C@     �E@      @@      7@     �@@      9@      0@      (@      3@      1@      1@      1@      0@      3@      "@      @      @      @      @      @      @      "@       @      @      @      �?      @       @      @               @              �?       @               @       @      �?              �?      �?      �?              �?              �?      �?              �?              �?              �?               @              �?              �?      �?      �?              �?      �?              �?      �?              �?      �?              �?      �?       @       @       @      �?      �?               @              @      �?      @      @      @      @      @      @      @      @      @      "@      @      "@       @      ,@      &@      (@      6@      0@      .@      5@      :@      ?@      3@      4@      2@      6@      C@      ;@     �B@      E@      8@      >@      F@     �C@     �@@      E@      A@      E@     �A@      A@      C@      F@     �@@     �A@      9@      A@      E@      7@      5@      ?@      =@      9@      1@      9@      5@      2@      .@      (@      "@       @      @      @      @      @      @      @      @      @      @               @      �?       @              �?      �?      �?      �?              �?        b#Eb0      y;l�	��-z���A/*�`

mean squared errorѹ=

	r-squared>��>
�E
states*�E	    >K�    �@    ��=A!c�Y�g��)�.:ށ��@2�"h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`����tO����f;H�\Q������%���9�e����K�����1���=��]����/�4��ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽K?�\��½�
6�����G�L������6���Į#�������/���EDPq��|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽�>�i�E������z5��!���)_��G-ֺ�І�̴�L����%�f*��8ŜU|�:[D[Iu�z����Ys��-���q�        �-���q=z����Ys=%�f*=\��$�=�/k��ڂ=̴�L���=����z5�=���:�=��s�=������=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=�EDPq�=����/�=�Į#��=���6�=G�L��=�d7����=�!p/�^�=(�+y�6�=�|86	�=��
"
�=���X>�=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>7'_��+/>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              @      "@      *@      2@      E@     �O@      c@     @k@     �|@     H�@     :�@     P�@     ��@     s�@    ���@    �3�@    �D�@    ���@    ���@    ���@     ��@     M�@     �@     ��@     ��@    ���@    ���@     ��@     ��@     W�@     ��@    �o�@     |�@    ��@    ���@     d�@    ���@     ��@     ��@    ��@    @��@    @��@     >�@     d�@    ���@     ��@    @s�@    �d�@     	�@    ���@    @��@    �C�@     Q�@    ��@     3�@    �i�@    �)�@    @��@    �/�@     ��@     �@    ���@     E�@    ���@     >�@     z�@    ���@    ���@    ��@     �@     �@     !�@     ��@     ]�@     �@     _�@     4�@     ث@     8�@     >�@     t�@     ��@     p�@     X�@     D�@     ؛@     ��@     �@     |�@     Г@     ��@     �@     ��@     ��@      �@     X�@     �@     ��@     0�@     ��@     �|@     py@      {@     Pu@     �v@     �s@     �q@     �p@     �o@     �j@      k@      h@     @d@      f@     `d@      a@     @\@      _@      ^@     �X@     �\@     @U@     �P@     �R@     @Q@     �P@      K@      N@      N@      K@     �I@      E@     �G@     �D@     �B@      ?@      =@      >@      =@      =@      =@      8@      =@      0@      9@      <@      (@      7@      .@      "@      &@      *@      .@      8@      $@       @      $@      *@       @      *@      $@      @      �?      @      @      @      @      *@       @      @      @       @      @      @      (@      @       @      @       @      @      @      @      @               @      @       @      @      @       @      @      @      �?      @      @       @      @      �?      @      �?              �?      �?      @      �?      @      �?      @       @       @              @      @      �?      �?       @       @              �?              �?      �?      @              �?       @      �?      �?      �?       @               @              �?      �?              @      �?      �?      �?       @      �?              �?              �?               @              �?      �?               @              �?               @      �?              �?      �?      �?      �?      �?              �?              �?              �?              �?              0@      &@      �?              �?      �?      �?              �?      �?              �?               @              �?              �?      �?               @              �?              �?              �?              @      @               @       @               @      @      �?              �?              �?       @      @              �?              @      �?       @               @      �?              �?       @      �?      @      �?      �?              @      �?      �?      @      @       @      @       @      @              @      @       @      @      @      @      �?      @      �?       @      @      @      �?              @      @       @      @       @      �?      @      @       @      @      @      @       @      @      @      @      @      $@      @      @      @      @      "@      @      @      @      &@      @      "@      @       @      $@      &@      "@      $@       @      .@      .@      &@      &@      $@      4@      1@      6@      .@      8@      7@      0@      3@      5@      :@      <@      6@      <@      9@      C@      :@      8@      B@     �F@      C@     �I@     �E@     �F@      F@      G@     @P@     @S@      O@     @R@      T@      T@      T@     �S@      W@     @_@      `@     @`@     �c@     �c@     �e@     �g@      g@     @i@     �k@      l@     @o@     `p@     0t@     �v@     �v@     �y@     �y@      @     ��@      �@     P�@     ��@     ��@     ��@     X�@     @�@     @�@     ��@     ��@      �@     h�@     �@      �@     �@     "�@     B�@     �@     H�@     ��@     X�@     L�@     ��@     հ@     ٲ@     c�@     �@     ��@     Q�@     ѽ@     b�@    ���@     ��@    ���@    �v�@      �@    ���@     ��@    @%�@    �F�@    ���@    @��@    �h�@    ���@    �V�@    @��@    ���@     ��@     _�@    �^�@     ��@    @�@     ��@    ���@    @K�@    �>�@    ��@    ���@    @�@    @k�@    ���@     ��@    @��@    ���@    �<�@    �f�@    ��@     �@    ���@    ��@    �m�@     `�@    ���@     "�@     t�@    �O�@    �<�@     ��@     |�@     ��@    ���@    ���@     ��@    �M�@     f�@     ��@     l�@     ��@     ��@     l�@     ��@     0s@     �b@     @X@      H@     �B@      0@      &@      @      @        
�
predictions*�	    �|ؿ   �J^@     ί@!  ��L:@)��kd��Q@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��vV�R9��T7�����d�r�x?�x��1��a˲���[���FF�G �I��P=��pz�w�7�����%ᾙѩ�-߾�*��ڽ�G&�$���f����>��(���>I��P=�>��Zr[v�>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@{2�.��@!��v�@�������:�              �?      �?      �?      �?       @       @      @      $@      1@      .@      7@     �A@      =@      ?@      G@      K@     �J@     �Q@      O@     �R@     �Q@      U@     @S@      S@      U@      Y@     �U@      Z@      X@     �V@      R@     �N@     �P@     �S@      O@      L@     �M@      J@      <@      E@     �D@      ?@      >@     �A@      9@      6@      1@      2@      7@      .@      ,@      .@      &@      $@      "@      (@      *@      @      "@       @      @      @       @      @      @              @              @      �?       @      �?       @       @      �?      �?              �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              @      �?              �?              �?              �?       @              @      �?      @      @       @      @      @              @      @      @      @      @      "@       @      �?       @      @      0@      0@      &@      ,@      7@      "@      2@      7@      7@     �A@      3@      A@      A@      B@      ?@      =@      @@      D@     �J@     �C@     �G@     �D@     �E@      J@     �K@     �O@      H@     �H@     �A@      C@      N@      B@     �D@      H@      A@      @@     �@@     �@@      B@      :@      @@      0@      7@      4@      $@      (@      *@      "@       @      @      @      @      @       @       @      @      �?       @      �?      @      @       @       @              �?              �?              �?              �?        �21      "���	�Uz���A0*�b

mean squared error��=

	r-squared`Q�>
�G
states*�G	   ��<�   ��h@    ��=A!�jK�a��) ݂]P:�@2�#h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���-�z�!�%�����i
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`����f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J���1���=��]����/�4��ݟ��uy�z������Qu�R"�PæҭUݽH����ڽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�ĽK?�\��½�
6������Bb�!澽5%����G�L������6��|_�@V5����M�eӧ�y�訥�V���Ұ����@�桽����z5��!���)_�����_����e���]����x����̴�L�����/k��ڂ�\��$��%�f*�z����Ys��-���q�        �-���q=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=�/k��ڂ=G-ֺ�І=�1�ͥ�=��x���=e���]�=���_���=!���)_�=����z5�=���:�=�>�i�E�=��@��=y�訥=��M�eӧ=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
6����=K?�\���=�b1��=��؜��=�d7����=�!p/�^�=��.4N�=;3����=(�+y�6�=�|86	�=��
"
�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�#              @      @      $@      4@     �G@     �S@      a@     p@     �@     P�@     ��@     ��@     }�@     ��@    ���@    �h�@    �"�@    �;�@    �.�@    �h�@    ��@     '�@     ��@     �@     ��@     ��@     ��@    ���@     ��@     ��@    �k�@     Y�@     ��@    ���@    �<�@    ���@     �@    �I�@    ���@     ��@    @��@     �@    @��@    �n�@     ��@    ���@     o�@     �@     ��@    ��@    ��@    ��@    @�@     ��@    ���@     '�@    @N�@    @�@     ��@     _�@    ���@    �u�@     ��@    ��@    ���@    ���@    ���@     ��@    �-�@     پ@     6�@     �@     �@     �@     6�@     �@     ��@     v�@     (�@     ֨@     V�@     �@     ��@     ȡ@     8�@     ��@     4�@     D�@     p�@     �@     |�@     ��@     ��@     X�@     �@     �@     (�@     H�@     �@     �@     �@      @     @{@     �y@     x@     pt@     pr@     �t@      q@     �o@     �k@      k@     �h@      j@     �e@     �a@     �c@     �b@     �^@      [@     �W@      V@     �[@     �W@     �R@      M@     �J@      N@      N@     �I@     �H@      ?@      C@     �G@     �H@      >@     �A@     �B@      :@      7@      =@     �A@      ?@      8@      .@      2@      0@      9@      5@      2@      *@      .@      3@      (@      "@      "@       @      "@      "@      @      @       @      @       @      @      @      "@      @      @      @      @       @      @      @      @      "@      @      $@      @      @      @      @       @       @      @       @      @       @               @      @      �?      @       @      �?       @      @       @      @      �?              @      �?       @      �?              �?              @      @      �?       @              �?       @              �?       @      �?              �?       @               @       @       @      @       @       @      �?       @      @               @              �?       @      �?       @              �?              �?       @      �?      �?              �?       @              �?      �?              @       @              �?              �?      �?      �?              �?       @              �?              �?      �?               @      �?      �?       @              �?       @      �?              �?      ,@      2@              �?              �?      �?      �?              �?      �?      �?      �?              �?      �?              �?              �?              �?      �?      �?      �?      �?              �?              �?              �?       @              �?      �?              �?              �?              �?      �?      �?               @      �?       @               @       @              �?              �?      �?       @      @      �?      @      �?              �?      �?      �?      �?      @              @      �?       @      �?      @      @       @      �?       @      @      @      �?       @      @               @      @      @      @      �?      @      @              @       @      @               @      �?      �?       @       @      �?      @      @      @       @      @      @      @      @      @      @      @      @      $@      &@       @      @      *@       @      "@      &@       @      @      "@      @      0@      @       @      .@      *@      0@      "@      0@      0@      $@      ,@      (@      3@      9@      9@      &@      0@      9@      8@      ;@     �@@      E@     �C@      6@     �@@      E@      >@     �I@      @@      E@      E@      K@      G@      G@     @Q@     �R@     @R@     �U@     �S@     �S@     �Y@     �[@     @\@      `@     �a@      _@     �b@      d@     �e@      f@     `h@     �j@     `l@     Pq@     ps@     �r@     �r@     �v@     @x@     �{@      |@     ��@     �@     X�@     ��@     h�@     ��@     ؉@      �@     ��@     Ԑ@     ��@     p�@     0�@     ��@     ��@     ȝ@     �@     V�@     Z�@     b�@     ��@     @�@     f�@     �@     ��@     ۱@     ��@     ��@     �@     ��@     Z�@     Ҿ@     Q�@    ��@     ��@    ���@     I�@     ��@     �@     ��@     ��@    ���@    @0�@    �
�@    ���@    ��@    ���@    �O�@    ���@     ��@    �V�@     �@    ���@    @��@    �(�@    ���@    @��@    �z�@    �4�@     *�@    �X�@    ���@    @��@     ��@    ���@    @/�@    ���@     ��@     ��@    ���@    ���@    ��@    ���@    ���@    ���@    �C�@     ��@    ���@    ��@    �E�@    ��@    �q�@     !�@    ��@    �}�@     `�@     S�@     v�@     ޵@     4�@     ��@     �@     ��@     pv@     @j@     �Y@     �M@      A@      "@      .@      &@       @        
�
predictions*�	   @T�ܿ   ���
@     ί@!  ���	@)�W튌�D@2���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9���d�r�x?�x��>h�'��6�]���1��a˲��FF�G �>�?�s�����~��¾�[�=�k���FF�G ?��[�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�P�1���?3?��|�?�E̟���?ܔ�.�u�?��tM@�Š)U	@u�rʭ�@�������:�              �?              �?              �?      �?      @      @      @       @      &@      0@      (@      9@      5@      9@      G@     �A@      H@     �J@     �Q@     �R@      T@      V@     �X@     �S@     �T@     �S@     @V@     @X@      R@      V@     �W@      S@      R@     �P@     @S@     �M@      O@      M@      I@     �B@      7@      D@      A@      9@      7@      9@      :@      9@      2@      4@      &@      3@      @      @      &@      "@      @      @      @      @      @      @      @      @      @      @      �?      �?       @      @      @       @       @       @               @       @       @              �?               @               @              �?      �?              �?              �?              �?              �?              �?               @      �?              �?      �?              @      �?              @      @      �?      @      �?       @      @       @      @      @      @      @      @       @      @      $@      @      "@      $@      @      @      @      "@      3@      (@      1@      6@      $@       @      3@      &@      2@      4@      9@      @@      <@      A@      D@     �B@     �C@      D@     �F@      G@      H@      N@      B@      K@     �I@     �H@     �N@     �E@     �G@     �G@     �C@      D@      G@     �@@      <@      B@      D@      B@      :@      7@      :@      <@      2@      .@      3@      3@      @      .@       @      $@      @      @       @       @      @      @      �?      @      �?      @      @      �?      @      �?              �?      �?              �?              �?        }��0      xiB	t�z���A1*�_

mean squared errorq=

	r-squared|"�>
�F
states*�F	   @�r�   ��@    ��=A!<�����)�h=e��@2�"�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U�H��'ϱS��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D�/�p`B�p��Dp�@�����W_>�p
T~�;�u 5�9��z��6��so쩾4�6NK��2�_"s�$1�7'_��+/��'v�V,����<�)�4��evk'���o�kJ%�4�e|�Z#���R����2!K�R���J��#���j�Z�TA[��RT��+��y�+pm�nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]����/�4��i@4[���Qu�R"�H����ڽ���X>ؽ��
"
ֽ�|86	Խ��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�Bb�!澽5%����G�L������6���Į#�������/��|_�@V5����M�eӧ�y�訥�V���Ұ���>�i�E��_�H�}������:������z5��e���]����x�����1�ͥ��G-ֺ�І��/k��ڂ�\��$��x�_��y�'1˅Jjw�z����Ys��-���q�        �-���q=z����Ys=�8ŜU|=%�f*=̴�L���=G-ֺ�І=���_���=!���)_�=��s�=������=_�H�}��=��@��=V���Ұ�=��M�eӧ=|_�@V5�=<QGEԬ=�8�4L��=�EDPq�=����/�=�
6����=K?�\���=�d7����=�!p/�^�=;3����=(�+y�6�=�|86	�=��
"
�=���X>�=H�����=PæҭU�=�Qu�R"�=i@4[��=z�����=ݟ��uy�=�/�4��==��]���=��1���='j��p�=��-��J�=�K���=�9�e��=����%�=f;H�\Q�=�tO���=�f׽r��=nx6�X� >�`��>�mm7&c>y�+pm>RT��+�>���">Z�TA[�>�#���j>�J>2!K�R�>��R���>Łt�=	>��f��p>�i
�k>%���>��-�z�!>4�e|�Z#>��o�kJ%>4��evk'>���<�)>�'v�V,>_"s�$1>6NK��2>�so쩾4>�z��6>u 5�9>p
T~�;>����W_>>p��Dp�@>/�p`B>�`�}6D>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�"              �?      @      @      *@      4@     �D@     �V@     @d@     �r@     `�@     4�@     ~�@     پ@     k�@     ��@    ���@     ��@     �@    ���@    ��@     ��@     s�@    �>�@     ��@    ��@     o�@     ��@    �D�@    �;�@     "�@    �o�@    �<�@     ��@     8�@    ���@    �L�@    �.�@    �7�@     x�@     f�@    ���@    ���@     5�@    @��@     ��@    �&�@     ��@     d�@    ���@    ���@     :�@    ���@    ��@     ��@    �*�@    �c�@    @~�@     }�@     i�@    ���@    �	�@     ��@     ��@     ��@    ��@    ���@    �=�@    �[�@     ��@     ��@     �@     ��@     ��@     �@     �@     ��@     (�@     ��@     4�@     ܩ@     ��@     D�@     ��@     ��@     j�@     ̞@     �@     И@     ��@     |�@     \�@     ��@     T�@     H�@     Ћ@     ��@     ��@     0�@     �@     ��@     @~@     @}@     �y@     �{@     pv@     �u@     `s@     �q@      r@     `l@     �m@     `h@      f@      f@     �e@      `@     @d@     �[@     @Z@     �^@     �Z@     �W@      X@      O@     @S@     @P@     �S@     �P@      K@      I@     �P@      K@     �@@      B@      D@     �A@      ;@      ?@      ;@      6@      =@      :@      8@      ?@      8@      6@      4@      4@      1@      1@      5@      4@      $@      @      &@      0@      &@      @      2@      "@      3@      (@      @      @      &@      "@      @       @      *@      @       @       @      @      *@      @      @      @      @      "@      @       @      @      �?      @      @      �?      @       @      @      @      @      @      @      @      @      @      �?       @      @       @       @       @       @               @      @      �?              �?      @      @      �?      @      @      �?      @       @       @      @       @      �?               @      �?              �?              @              �?      �?              �?      �?              �?      �?      �?               @              �?              �?      �?       @               @              �?              �?              �?      �?       @              �?              �?              �?              �?              �?              �?              �?              �?              �?      *@      @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?               @              @               @       @       @              �?       @              �?      @       @      �?      �?       @               @              �?      �?      �?               @      �?              @      @      �?      @      �?      �?       @       @               @              @      @      @      �?      �?      �?               @              @      @      @      �?       @      @      �?       @       @      @      �?      @              @       @      @              @      �?       @       @      �?      @      @       @      @      @      @      @      @      @       @      "@      @      @      @      (@       @      @      @      @      @      @      $@       @      "@       @      "@      (@      &@      @      @      $@      @      0@      &@      0@      @      ,@      $@      .@      ,@      (@      2@      3@      0@      2@      7@      5@      B@      2@      <@      8@      ?@      8@      C@      >@      D@     �B@     �C@      E@     �F@     �M@     �J@      R@     �N@     �K@     �S@     �R@      V@     �T@      Y@      [@      [@     �^@     �]@     @a@     �a@     �`@     �`@     �d@      i@     @k@     @i@     �k@     �m@     �p@     0r@     �t@     @x@     �w@     �z@     �}@     �@     �@     ��@      �@     ��@      �@     ��@     p�@     ȍ@     \�@     ��@     �@     $�@     ��@     <�@     �@     ԟ@     B�@     ��@     Τ@     �@     >�@     V�@     T�@     v�@     K�@     1�@     �@     B�@     k�@     3�@     C�@     ��@    ���@    ���@    ��@     	�@     <�@    �0�@    �u�@     ��@    @��@    ���@    ���@    @��@    �j�@     ��@    �Q�@    � �@    @�@     ��@    �7�@    �q�@    @��@    ���@     �@    @�@    @��@    @��@    @d�@     ��@     ��@    ���@    ���@    @��@    @�@    �G�@    ���@    ���@     ��@    ���@    �*�@     2�@    �b�@    ���@    �?�@    ���@     1�@    ���@     ��@     ��@    �X�@    ��@     k�@     �@     ͽ@     ;�@     e�@     2�@     ��@     ��@     @�@     �y@      l@      a@      P@      B@      .@      @      @       @        
�
predictions*�	   ���ؿ   `M�@     ί@!   p� �)�^�K��O@2��^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.���5�i}1���d�r�>�?�s���O�ʗ������%�>�uE����>�FF�G ?��[�?��d�r?�5�i}1?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?S�Fi��?ܔ�.�u�?��tM@�6v��@h�5�@!��v�@زv�5f@�������:�              �?      �?              @              @      @      @      *@      @      "@      6@      ;@      7@      A@      L@     �H@     @Q@     �S@     �W@      X@      \@     @\@     @Y@      Y@     �W@      Y@     �T@     @Y@     �U@      X@     @V@     �S@      S@     @S@      N@      M@      H@      I@     �C@      E@      @@      F@      <@      7@      *@      *@      *@      9@      "@      4@      3@      &@      .@      @      ,@      0@      @      @       @      @      @              @      *@      @      @      �?      @       @       @      @       @       @      @       @              �?      �?      @              �?              �?               @              �?              �?              �?              �?              @              �?      �?      @      �?      �?      �?       @      �?      �?      �?       @      @      @      @              @       @      @      @      $@      $@      &@      "@       @       @      @      "@      @      3@      0@      1@      9@      2@      7@      3@      @@      6@      @@      >@     �A@     �B@      ?@      B@      C@     �A@      C@      F@      F@     �F@      C@     �C@     �D@      A@     �B@     �A@      H@      ;@     �E@      =@     �D@     �E@      6@      7@      9@      <@      9@      ,@      ,@      7@      ,@      &@       @      @      $@      @      @      @       @      �?      "@              @      @      @      �?      �?              �?      �?              �?      �?              �?      �?              �?              �?        Oz�