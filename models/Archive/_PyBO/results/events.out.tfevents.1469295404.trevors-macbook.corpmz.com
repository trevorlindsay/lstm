       �K"	   ����Abrain.Event:2_G��|�     �.y*	J����A"��

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
 *9{9?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"�  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ;*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��*
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
 *9{9?*
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
valueB"p  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ;*
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
 *9{9?*
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
valueB"p  �  *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *�ħ;*
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
 *9{9?*
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
 *�ħ�*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *�ħ;*
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
 *�ħ�*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *�ħ;*
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
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
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
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
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
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_8Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��
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
��*
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
:��"	�q�