       �K"	   S���Abrain.Event:2��N|�     �.y*	7�S���A"��

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
 *ѿ?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB"�    *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *��&�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
valueB
 *��&=*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��
�
6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
 *ѿ?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB"�    *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *��&�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
valueB
 *��&=*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
 *ѿ?*
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
��*
	container *
shared_name * 
_output_shapes
:
��
�
Ymodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shapeConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB"�    *
_output_shapes
:
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/minConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *��&�*
_output_shapes
: 
�
Wmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/maxConst*
dtype0*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
valueB
 *��&=*
_output_shapes
: 
�
amodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/RandomUniformRandomUniformYmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/shape* 
_output_shapes
:
��*
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
��
�
Smodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniformAddWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/mulWmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform/min*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
�
?model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AssignAssign8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatrixSmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Initializer/random_uniform*
validate_shape(*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking(*
T0* 
_output_shapes
:
��
�
=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/readIdentity8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMulMatMul8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasVariable*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Hmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Initializer/ConstConst*
dtype0*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
valueB�*    *
_output_shapes	
:�
�
=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AssignAssign6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasHmodel/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Initializer/Const*
validate_shape(*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking(*
T0*
_output_shapes	
:�
�
;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/readIdentity6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
�
.model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/addAdd8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
 *ѿ?*
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
 *��&�*
_output_shapes
: 
�
,model/dense_w/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_w*
valueB
 *��&=*
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
 *��&�*
_output_shapes
: 
�
,model/dense_b/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@model/dense_b*
valueB
 *��&=*
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
��*
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
��
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
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
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
��*
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
��
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
:�
�
Tmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMulMatMulKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b(*
transpose_a( *
T0* 
_output_shapes
:
��
�
Vmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1MatMul8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concatKmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape*
transpose_b( *
transpose_a(*
T0* 
_output_shapes
:
��
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
��*
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
��
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
:�
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
��
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
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_0Identitymodel/clip_by_global_norm/mul_1*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_2MulMmodel/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_1Identitymodel/clip_by_global_norm/mul_2*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_3MulVmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_2Identitymodel/clip_by_global_norm/mul_3*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_4MulMmodel/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_3Identitymodel/clip_by_global_norm/mul_4*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
model/clip_by_global_norm/mul_5MulVmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1model/clip_by_global_norm/mul*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
6model/clip_by_global_norm/model/clip_by_global_norm/_4Identitymodel/clip_by_global_norm/mul_5*i
_class_
][loc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
��
�
model/clip_by_global_norm/mul_6MulMmodel/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1model/clip_by_global_norm/mul*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
�
6model/clip_by_global_norm/model/clip_by_global_norm/_5Identitymodel/clip_by_global_norm/mul_6*`
_classV
TRloc:@model/gradients/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/add_grad/Reshape_1*
T0*
_output_shapes	
:�
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
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
f
model/zeros_8Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
\
model/zeros_9Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_10Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
g
model/zeros_11Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_12Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_13Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_14Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
g
model/zeros_15Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Cmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AdamVariable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Hmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam/readIdentityCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
g
model/zeros_16Const*
dtype0*
valueB
��*    * 
_output_shapes
:
��
�
Emodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1Variable*
	container * 
_output_shapes
:
��*
dtype0*
shape:
��*K
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
��
�
Jmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1/readIdentityEmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
T0* 
_output_shapes
:
��
]
model/zeros_17Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Amodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AdamVariable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Fmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam/readIdentityAmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
]
model/zeros_18Const*
dtype0*
valueB�*    *
_output_shapes	
:�
�
Cmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1Variable*
	container *
_output_shapes	
:�*
dtype0*
shape:�*I
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
:�
�
Hmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1/readIdentityCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
T0*
_output_shapes	
:�
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
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_1*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_2*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_3*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
�
Tmodel/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/ApplyAdam	ApplyAdam8model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatrixCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/AdamEmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_4*K
_classA
?=loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix*
use_locking( *
T0* 
_output_shapes
:
��
�
Rmodel/Adam/update_model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/ApplyAdam	ApplyAdam6model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/BiasAmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/AdamCmodel/model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/Adam_1model/beta1_power/readmodel/beta2_power/readmodel/Variable_1/readmodel/Adam/beta1model/Adam/beta2model/Adam/epsilon6model/clip_by_global_norm/model/clip_by_global_norm/_5*I
_class?
=;loc:@model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias*
use_locking( *
T0*
_output_shapes	
:�
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
��
�
0model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell0/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell1/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
��*
T0*
N
�
:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMulMatMul:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/concat=model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Matrix/read*
transpose_b( *
transpose_a( *
T0* 
_output_shapes
:
��
�
0model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/addAdd:model_1/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/MatMul;model/RNN/MultiRNNCell/Cell2/BasicLSTMCell/Linear/Bias/read*
T0* 
_output_shapes
:
��
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
:��"	d����!      ����	�T�S���A*�C

mean squared error��S=

	r-squared�,��
�6
states*�6	   �p��   ��@   �$[RA!NuF�1�K�)yG
�A2��Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��h���`�8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%ᾙѩ�-߾E��a�Wܾ�iD*L�پ�_�T�l׾��>M|Kվ��~]�[Ӿjqs&\�ѾK+�E��Ͼ['�?�;;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��~��¾�[�=�k���*��ڽ�G&�$��5�"�g���0�6�/n���u`P+d����n�����豪}0ڰ���������?�ګ�;9��R���5�L�����]������|�~���MZ��K���u��gr��R%������39W$:���.��fc���X$�z��
�}�����4[_>������m!#���
�%W����ӤP�����z!�?��T�L<��u��6
��K���7��[#=�؏�������~�f^��`{�E'�/��x��i����v��H5�8�t�BvŐ�r�ہkVl�p�w`f���n�=�.^ol�ڿ�ɓ�i�:�AC)8g�cR�k�e������0c�w&���qa�d�V�_���u}��\�4�j�6Z�Fixі�W���x��U��
L�v�Q�28���FP�������M�6��>?�J���8"uH���Ő�;F��`�}6D��so쩾4�6NK��2����<�)�4��evk'��tO����f;H�\Q���Qu�R"�=i@4[��=�f׽r��=nx6�X� >4��evk'>���<�)>6NK��2>�so쩾4>p
T~�;>����W_>>p��Dp�@>��Ő�;F>��8"uH>6��>?�J>������M>28���FP>�
L�v�Q>H��'ϱS>��x��U>Fixі�W>4�j�6Z>��u}��\>d�V�_>w&���qa>�����0c>cR�k�e>:�AC)8g>ڿ�ɓ�i>=�.^ol>w`f���n>ہkVl�p>BvŐ�r>�H5�8�t>�i����v>E'�/��x>f^��`{>�����~>[#=�؏�>K���7�>u��6
�>T�L<�>��z!�?�>��ӤP��>�
�%W�>���m!#�>�4[_>��>
�}���>X$�z�>.��fc��>39W$:��>R%�����>�u��gr�>�MZ��K�>��|�~�>���]���>�5�L�>;9��R�>���?�ګ>����>豪}0ڰ>��n����>�u`P+d�>0�6�/n�>5�"�g��>G&�$�>�*��ڽ>�[�=�k�>��~���>�XQ��>�����>
�/eq
�>;�"�q�>['�?��>K+�E���>jqs&\��>��~]�[�>��>M|K�>�_�T�l�>�iD*L��>E��a�W�>�ѩ�-�>���%�>�uE����>�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�             �d@     \�@     ��@     T�@     �@     Ƨ@     ��@     q�@     &�@     '�@     G�@    ���@     (�@    �|�@    @��@    ���@    @��@    ���@    @:�@     ��@    ���@    �R�@    @��@    @��@    �N�@     [�@     ��@     ��@    ���@    ���@    ��@    ���@    ���@    `,�@     f�@     ��@    ���@    `b�@    @��@    @��@     q�@     �@    �y�@     ��@    @M�@    ��@    @�@    �E�@     �@    ��@    @��@    ��@    �i�@    �p�@    �b�@     "�@     �@    ���@    ���@    @y�@    �b�@    @��@    ���@    @2�@     b�@    ���@    @O�@    ���@    ���@    @i�@     :�@     ��@     ��@    ���@     ��@    �[�@     z�@    ��@    ���@     \�@     �@     ^�@     	�@     B�@     �@     {�@     I�@      �@     ��@     �@     x�@     d�@     ��@     ��@     2�@     ��@     8�@     ܚ@     Ę@     Ȗ@     @�@     ��@     ��@     �@     P�@     x�@     ��@     �@     ��@     X�@     @�@     Ѐ@      }@     @{@     z@     �y@     �w@     �t@     `t@      q@     �p@     �m@     �i@     `k@     �g@     �g@     �d@     �b@     �`@     �^@     �]@     �[@     �[@     �X@     �U@     @W@      U@      N@      T@     �O@      G@      I@     �H@     �H@      L@     �C@     �A@      @@      ?@      =@      ;@     �@@      >@      9@      9@      1@      1@      1@      "@      $@      (@      @      0@      $@      (@      *@      $@      .@      @      @      @      @      @      @      @      @      �?      @      @      @       @      @      �?      �?      �?      �?      �?       @      �?       @      �?      �?               @      @      �?      @      �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @      �?      @              �?       @               @       @      @      @       @       @       @      �?      �?       @      @      @      "@      $@      @       @      @       @      @      $@      $@      @       @       @      "@      $@      "@      (@      $@      $@      *@      2@      1@      ,@      3@      *@      1@      @@      A@      ?@      B@      A@      E@      ;@      =@     �D@     �H@     �N@      H@     �I@     �N@     �Q@     �Q@     @S@     �Q@      O@     �U@     @V@     �W@      Y@     @[@     �\@     �[@     �`@     �b@      d@     @d@     @f@     �f@     �j@     �l@     �o@     �n@     pq@     @s@     �t@     v@     �z@     �|@     �z@     �}@     @@     �@     Ђ@     �@     �@     H�@     H�@     ȋ@     ��@      �@     Б@     �@     �@     ĕ@     ��@      �@     ��@     ��@     ��@     T�@     �@     ޥ@     ��@     Z�@     ��@     Z�@     ��@     ��@     ��@     ȴ@     ��@     ׹@     �@     ֽ@    ��@     3�@     ��@    ���@    ���@    �9�@     ��@    �
�@     �@    ���@    ���@     �@     ��@    @�@    @��@     U�@    @z�@    ���@    ���@    ��@    @}�@    ���@     ��@    `��@    ���@    `��@     ��@    ���@    `��@     ��@    ���@     0�@     %�@    ���@    `X�@      �@    `��@    �7�@    `��@    ��@    `n�@     ��@    �p�@     
�@    `�@     u�@     (�@    `��@    �=�@    `:�@    @,�@    �1�@    �i�@    �N�@     S�@     ��@     ��@    ���@    ���@    ���@     �@    ���@     ��@    ���@     )�@     b�@     ��@    @��@     ��@     �@     մ@     {�@     @�@     
�@     �@     ,�@     ��@     ��@     X�@     �k@        
�
predictions*�	   �%�v�   �ٚ�?     ί@!  'sh�l@)�.�j�.@2�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�P}���h�Tw��Nof����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G��!�A����#@�d�\D�X=���%>��:�uܬ�@8��u�w74���82���bȬ�0���VlQ.��.����ڋ�jqs&\�ѾK+�E��ϾI�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?uܬ�@8?��%>��:?d�\D�X=?a�$��{E?
����G?�qU���I?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�������:�              �?              �?              �?              �?              �?      �?       @      �?      �?      �?               @               @              �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              @              �?      �?              �?      �?               @      @       @              @      �?      $@      @      @      @      "@       @      @      @      0@      ,@      .@      (@      &@      *@      "@      (@      @      @      .@      ,@     �@@      6@     �E@     �B@      G@      I@     �M@      B@     �O@     �L@      R@     �]@      b@     �i@      n@     Pp@     �r@     �v@     �w@     �w@     px@     @p@     `f@     @R@      (@      @        �~�ǒ.      F<�	͓;T���A*�]

mean squared error�.D=

	r-squared �
�G
states*�G	   �R�   �x�@   �$[RA!f�}O��)9�H��A2�#h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�k���f��p�Łt�=	���R����2!K�R���J��#���j�Z�TA[�����"�RT��+��y�+pm��mm7&c��`���nx6�X� ��f׽r����tO����f;H�\Q������%���9�e����K��󽉊-��J�'j��p���1���=��]���ݟ��uy�z�����i@4[���Qu�R"�PæҭUݽH����ڽ���X>ؽ��
"
ֽ�|86	Խ(�+y�6ҽ;3���н��.4Nν�!p/�^˽�d7���Ƚ��؜�ƽ�b1�Ľ5%����G�L�������/���EDPq���8�4L���_�H�}��������嚽��s���/k��ڂ�\��$���-���q�        �-���q=:[D[Iu='1˅Jjw=�8ŜU|=%�f*=\��$�=̴�L���=G-ֺ�І=�1�ͥ�=e���]�=���_���=!���)_�=����z5�=_�H�}��=�>�i�E�=��@��=V���Ұ�=y�訥=<QGEԬ=�8�4L��=�EDPq�=����/�=�Į#��=���6�=G�L��=5%���=�Bb�!�=�
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�#             �^@     ��@     0�@     t�@     ��@     �@     ��@     ڧ@     \�@     ��@     ݽ@     I�@     ��@     ��@     �@     w�@     B�@     ��@     ��@    @+�@     (�@    @,�@    �p�@     �@      �@    @��@     ��@    @��@    ��@    �.�@    ���@    ���@    �Y�@     ��@    @��@    � �@     ��@    `h�@    �
�@    �F�@    `��@    �_�@    p
�@    P��@    �i�@    ���@    P�@    q�@    У�@    Ь�@    ��@    �l�@    p��@    `��@     1�@    ���@    `��@    @�@    ���@    ���@     ��@    ���@     N�@    @��@     �@     ��@     �@    @��@    �8�@     e�@     }�@    ���@    ���@    �x�@     ��@    �1�@     �@     �@     2�@     ޹@     -�@     w�@     �@     Ų@     ��@     �@     ��@     r�@     ��@     �@     t�@     @�@     ��@     �@     X�@     ��@     �@     T�@     �@     ��@     ��@     <�@     H�@     ��@     x�@     ��@     x�@     d�@     �@     ��@     0�@     Ȍ@     (�@     x�@     ��@     ��@     �@     ��@     0�@      �@     ��@     h�@     ��@     ��@      @     �}@     �y@     �{@     �{@      y@     `y@     u@     �u@     u@     �u@     �t@     0q@     @q@     0p@     @o@     �o@     �n@     �j@     �i@     �k@     @h@     �e@     @g@     @g@     �d@     �c@     �b@     @c@      b@     �a@     �c@     �`@      ^@      `@     �\@     @^@      Z@     @X@      W@     @S@      V@     �V@     @Q@     @T@     �P@     �Q@     @S@     �J@      Q@     �G@      P@     �H@     �J@      K@     �J@      C@     �C@     �D@      C@      =@      ?@     �A@      ?@      ?@      8@      A@      2@     �A@      :@      =@      6@      8@      3@      6@      5@      1@      5@      ,@      .@      $@      &@      3@      7@      (@      &@      (@      $@      (@      @      $@      $@      @      @      @      $@      @      @      @      @      @      �?      �?      @      @      @      @       @      @      "@              @       @       @      @      @      �?      @       @      �?      �?      �?              �?      @      �?      �?              @       @       @      �?       @              �?              �?      �?              �?              @      �?               @      �?              �?              @      @              �?              �?      �?              �?       @              �?              �?               @              �?       @              �?              �?              �?              @      �?              �?      �?               @              �?       @       @      �?       @              �?       @              �?      @      �?      @              @      @      @      @      @      @      @       @      @      @      @      @      @       @      @      @      "@      @      @      "@      "@      �?       @      @      @      @       @      $@      *@      "@      (@      &@      2@      &@       @      "@      .@      $@      *@      .@      &@      4@      *@      .@      ,@      8@      2@      7@      .@      2@      3@      9@      9@      9@      :@      :@     �A@      :@     �C@     �D@      ?@     �G@     �F@      G@     �B@      E@      E@     �C@      I@      M@      L@     �O@      H@      K@     �T@      Q@     �S@     �S@     �P@      R@     �W@     @T@     @V@      Z@     �X@     �T@     �Z@     �\@     �_@     @^@      `@     �`@     @d@     �d@     `f@     �b@     �e@     �g@     �h@     @h@      k@     `l@     �m@     �n@      q@     0p@     �o@     �r@     r@     `s@     @u@     ps@     �w@     x@     Px@     �v@      {@     �|@      �@     �}@     �@     ؂@     0�@     (�@      �@     ��@     ��@     X�@     ؆@     `�@     P�@     X�@     ��@     X�@      �@     Б@     ؑ@     ��@     `�@     �@     ȕ@     ��@     ̗@     ��@     X�@     l�@     ��@     ȝ@     p�@     �@     (�@     :�@     `�@     �@     ~�@     j�@     ��@     .�@     ֬@     �@     L�@     f�@     W�@     ��@     ��@     Ҷ@     ��@     ��@     "�@     ��@    ���@     ��@    ���@    ���@     ��@     ��@    ���@     ��@    @�@    ���@    @3�@     ��@    ���@    @(�@     ��@    `�@    ��@    ���@    ���@    ���@     8�@    �'�@    �-�@    @��@    pz�@    ���@    @!�@    ��@    ���@     ��@    ��@    0B�@    ���@    �Q�@    @��@    ���@     ��@    @��@    �D�@    �A�@    �*�@    @'�@     G�@    �M�@    @��@     ��@    �F�@    �=�@    @K�@    @��@    @�@     %�@    ���@     ��@     c�@     O�@     �@     ��@     �@    ���@    �c�@     ��@    ���@     ��@     ��@     ��@     2�@     F�@     \�@     �@     ��@     d�@     �@     ��@     Ј@     �i@        
�
predictions*�	   �IÒ�   @a�|?     ί@!  �&6"�)c���8�?2�
�Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F��8K�ߝ�a�Ϭ(���(��澢f���侮��%ᾙѩ�-߾E��a�WܾK+�E��Ͼ['�?�;
�/eq
Ⱦ����ž�*��ڽ�G&�$�����?�ګ�;9��R��cR�k�e>:�AC)8g>�ѩ�-�>���%�>�f����>��(���>a�Ϭ(�>8K�ߝ�>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?�������:�
               @      �?      �?      @      *@      2@      D@      H@     @S@     �U@     �_@     @_@      b@      c@      b@      d@     �^@     �d@     @\@     �^@     �]@     @^@     @[@     �U@     �T@      T@     �Q@     �N@     �O@      O@     �J@     �M@     �C@     �C@      B@      7@      4@      7@      2@      5@      8@      3@      $@      &@      1@      (@      "@      @      &@      $@      @      @      @      @      �?      @      �?      @      @      @      @       @      @      �?      �?      �?               @      �?      �?      �?      �?       @       @              �?              �?              @      @              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              @      �?       @      �?      �?       @       @      �?      @      �?       @      @      @      @      @       @      @      "@      @      @      @      &@      @      @      &@      @      (@      0@      $@      5@      2@      5@      0@      9@      8@      =@      4@      <@      <@      @@     �@@     �B@     �A@      A@      E@     �E@      G@      B@     �D@     �B@      E@      H@     �C@      =@      3@      2@      ,@       @      @      @        M�=21      "���	Y��T���A*�b

mean squared errorbB=

	r-squared�f<
�L
states*�L	   ����   �Y@   �$[RA!o���w��)�>���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              @     @o@     ��@     ��@     ��@     ܗ@     ��@     f�@     N�@     ʬ@     !�@     o�@     p�@    �Z�@     p�@    �H�@    ���@    �t�@    ���@     ��@    ��@     w�@     ~�@    ��@    @��@    ���@    ���@    ���@    @��@    �|�@     &�@     ��@    ���@    @��@    ���@    ���@     ��@    `L�@    `��@    ���@    ���@    �)�@    ���@    �Q�@    ���@    �i�@     r�@    @4�@     ��@    ���@    �)�@    @��@    �)�@    `>�@     s�@    `�@    `@�@    ��@    @��@     ��@    ���@    `R�@    ���@    ���@    @��@    @H�@    ���@     ��@    ���@    @��@    @��@    ���@    @�@    ���@    ��@     C�@     ��@    �[�@    ���@    ���@    ���@    �j�@     �@     ��@     T�@     ��@     �@     ��@     ��@     ĳ@     ��@     \�@     ��@     �@     f�@     �@     D�@     ~�@     �@     �@     ,�@     `�@     �@     <�@     j�@     ��@     ��@     ĝ@     P�@     X�@     ș@      �@     ,�@     l�@     <�@     ,�@     (�@     p�@     p�@     t�@     P�@     t�@     $�@     ��@     X�@     ��@     �@     ��@     ��@     x�@     ��@     ��@     p�@     ��@     ��@     0�@     ��@     ��@     ؁@     ��@     X�@     �~@      �@     `~@     �{@     @|@     �z@     `{@     w@     Px@     pw@     �v@     0u@     �u@     Pt@     �w@     @t@      s@     `p@     �q@     �p@     �o@     @n@      m@     @l@     �i@      j@     `i@     @i@      e@     @i@     @g@     �d@     �b@     �`@     �`@      d@     �b@     �`@     �^@     �]@     �^@     �^@      \@     �Y@     @U@      Y@     �Y@     �^@     �T@      Z@      \@     @Y@      T@      U@      V@     �Q@      T@     �P@     �P@     @P@     �K@      O@     �N@     @P@      H@     �E@     �I@     �E@     �D@     �C@     �H@      A@     �A@      D@     �H@     �B@      =@      ?@      ?@     �@@      ;@      ?@      :@      7@      >@      3@      4@      2@     �@@      3@      1@      5@      9@      (@      .@      2@      6@      ,@      1@      0@      @      ,@      3@      (@      .@      $@      *@      @      (@      $@       @      @      @      (@      @       @      (@      $@      @      @      @       @      @       @      @      @      @      @      @      @      @              @       @      @      *@      @      @      @      @      �?       @       @      @      @      @      @      �?      �?      @      @      @      @              �?              �?       @      �?       @              O@     �Q@       @      @       @               @      �?      @      @      @      @      @       @      @      @      @       @      @      @       @      @       @       @      @      @      @      @      @      �?      @      @       @      @       @      @      "@      @      @      @      @      @       @      @      (@      $@      "@      @      $@      $@      &@      *@      .@      (@      .@      ,@      3@      ,@      "@      1@      .@      5@      .@      2@      ,@      5@      5@      5@      &@      ,@      4@      9@      2@      7@      :@      C@      9@      B@      ;@      <@     �@@     �A@     �A@      ?@     �B@      B@     �D@      A@      G@     �I@      G@     �J@     �H@     �C@     �G@     �L@      K@      O@     �M@     �P@     @R@      R@      S@     @Q@     @R@     �V@     �U@     �U@     @S@      U@     @Y@     @X@     �Z@     �Z@      Y@     �]@     �\@     �`@      ]@     ``@      `@     �b@     ``@     @`@     `c@     �f@     �g@     �d@     �b@     `d@     �g@     �f@      g@     �h@     �k@     @i@     @h@     �k@     �n@     �m@     �n@     `p@     �p@     �p@      u@     �s@     pr@     �s@     �t@     `r@     @t@     �v@     �y@     �w@      x@      }@     �~@     Pz@      @     �@     @     �~@     ��@     x�@     @�@     H�@     �@     h�@     X�@     Ȅ@     �@     ��@     (�@     @�@     ��@     Ȋ@      �@     ��@     8�@     �@     ��@     4�@      �@     �@     $�@     ��@     \�@     ��@     ��@     `�@     l�@     x�@     ��@     ��@     �@     Ԛ@     ț@     h�@     T�@     n�@     �@     H�@     ~�@     �@     ��@     ��@     H�@     ��@     �@     �@     ��@     ��@     ��@     t�@     &�@     ^�@     ��@     ��@     ��@     ��@     ��@     �@     )�@     8�@    ���@    ���@     T�@     ��@    ���@     �@    ���@    �l�@    ���@    �{�@    ���@     )�@     ��@    �H�@     �@    @	�@    @g�@    @��@     �@     P�@    `7�@    ���@    ���@    ��@    `��@    ���@    ���@     o�@     ��@     ^�@    �^�@    �t�@    �y�@     h�@     �@    `u�@    ��@    `��@    ��@    ��@    ���@     ��@    @��@    �=�@    `�@    ���@    � �@    ��@    ��@    ���@    ���@    @��@    ���@      �@    ��@    ���@     "�@     5�@     ��@    ���@    ���@     �@     ��@     R�@    ���@    ��@     1�@    ���@     )�@     >�@     N�@     �@     ��@     ��@     �@     (�@     ��@      m@      2@      �?        
�
predictions*�	   `V¿   �u�?     ί@!  ��=F�)4�:hj�@2�
�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�1��a˲���[��8K�ߝ�>�h���`�>����?f�ʜ�7
?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�������:�
              �?      @      @      "@      &@      =@     �C@      J@     �N@      L@     @P@     �U@      U@     @U@     �X@     �S@     �U@      [@     @R@     �T@     �Q@      Q@     �S@     �P@      L@     �M@      E@      I@      F@     �@@      D@      E@     �@@      5@      ;@      ;@      0@      9@      4@      ,@      &@       @      $@      ,@       @      "@       @      $@      @      @      @      $@      @       @      "@      @      @      @      @       @      @               @      �?      @      @       @      �?      �?       @      �?      �?              @              �?              �?      �?       @               @              �?              �?              �?              �?      �?       @              �?      �?              �?              @      @      @              �?      �?      @      �?       @      @      @       @      @      @      @      @      @       @       @      ,@      @      @      $@      (@      1@      .@      &@      2@      1@      4@      8@      4@      7@      :@      8@     �F@      A@      F@      F@     �B@     �M@      K@      O@      R@     �Q@     �R@     @U@      T@     �Q@     �S@     �S@     �R@     @T@     @V@     �M@     �P@      Q@      M@     �N@      :@      1@      $@      @      @      �?      �?      �?        A�>~B0      y5^�	@P"U���A*�`

mean squared error��I=

	r-squared�p�
�L
states*�L	   �;��   ��@   �$[RA!��"����)0}����A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              A@     Ps@     Pz@     �@      �@     ̔@     0�@     ܜ@     Ģ@     h�@     ��@     �@     ��@    ���@    ���@    ��@    ���@    �
�@    ��@     �@    �_�@    ���@    �"�@    ���@    ���@    �v�@     r�@    �&�@    ���@     �@    ���@     k�@     ��@    �F�@    ���@     �@    �^�@     "�@    �d�@     N�@    ���@     �@    @��@    `�@     ��@     ��@    ���@    ���@     z�@     ��@    �/�@     F�@    ���@    �e�@    �H�@    ���@    �w�@    `�@     ��@    �]�@    `*�@     �@    @^�@     �@     *�@    ��@    ���@    �p�@    ���@     U�@    ��@     ��@    ���@    ���@     ��@    ��@     l�@    ���@     ��@     h�@     I�@    ��@     w�@     ��@     f�@     /�@     Ϸ@     [�@     ��@     B�@     ��@     1�@     �@     ��@     ��@     ��@     &�@     �@     $�@     ި@     P�@     �@     Υ@     ĥ@     ��@     (�@     ��@     �@     ��@     V�@     Z�@      �@     ��@     X�@     ��@     �@     ț@     ؙ@     ��@     ��@     0�@     ��@     \�@     p�@     ȕ@     ��@      �@     �@     P�@     ��@      �@     @�@     4�@     8�@     X�@     l�@     ��@     ��@     ��@     ��@     H�@     ��@     P�@     `�@     ��@     ��@     Ї@     ��@     ؅@     ��@     �@     �@     8�@     ��@     Ѐ@     ��@     ��@      �@     p�@     ��@     `}@     �|@     Py@     �{@      {@     �w@     �y@      y@     �v@     �u@     Pv@     0u@     �v@     t@     `u@      q@     �r@      p@      r@     �p@      o@     �p@     @l@     �l@      l@     �i@     �j@     `g@      j@     �g@      c@      g@      c@     �e@     `e@      g@     `c@     �b@      c@     �d@     �a@     ``@     �_@      Y@     @]@     �\@     @X@     �_@     �]@     �V@      U@     �X@     �T@     �R@      W@     �S@      P@     �R@     @P@     @U@     �S@     @P@      R@     �Q@      M@     �P@     @Q@      H@     �O@      L@      H@     �H@      I@      F@      D@     �E@     �A@     �G@      >@     �B@      A@      E@      D@      B@     �C@     �A@      =@      3@      ;@      7@      3@      8@      ;@      6@      8@      ;@      6@      ;@      4@      .@      0@      *@      &@      (@      @      ,@      (@      1@      9@      "@      .@      0@      @      $@      1@      .@      0@      "@      1@       @      $@      @      @      @      @      @      @      @       @      �?      @       @      @      "@      @      @      @      @      @      @      @      �?     �Z@     `h@      @      @      "@       @      �?      .@      $@      0@      "@      @       @       @      @      *@      @      @      "@       @      "@      "@      1@      ,@      1@      *@      $@      &@      .@      $@      .@      0@      3@      *@      7@      0@      3@      =@      4@      9@      =@      3@      7@      =@      8@      6@      6@      3@      8@      6@      E@      A@      @@      A@      E@     �D@     �@@      =@      ?@      D@      C@      L@      ?@      I@     �E@      K@      K@      I@     �P@      N@      D@     @T@     �N@     �O@     �N@     �M@     �I@     �Q@     @R@     �Q@      P@     @P@     �T@     �T@      V@     �U@     �Y@     �V@      V@     �Z@      W@     �\@      `@     �[@     �^@      [@     �a@     �\@     `b@     �`@     �a@      b@     �a@      e@      b@     �e@     �c@     �b@      c@     @d@     �f@     �g@      g@     �l@     @h@     �j@      f@     �i@     �o@      n@      l@     �l@     �n@     pp@     `m@     pp@     �o@     �r@     �p@     �t@     �t@      u@     `u@     Pv@     �y@     0u@     �t@      x@     z@      |@     �y@     Py@     px@      z@      @     @z@     �}@     �~@     �@      �@     ��@     h�@     (�@     X�@     Ȃ@     ȁ@     `�@     ��@     ��@     ��@     �@     ��@     P�@     ��@     `�@     �@     `�@     ��@     ��@     ��@     �@     ��@     H�@     Ȏ@     �@     X�@     �@     ��@     ��@     l�@     ̒@     ��@     0�@     l�@     �@     h�@     ��@     `�@     ,�@     �@     |�@     Ț@     ��@     |�@     H�@     ��@     V�@     ~�@     �@     �@     ��@     0�@     v�@     �@     Φ@     V�@     d�@     D�@     ��@     ��@      �@     �@     |�@     e�@     �@     ��@     Ӵ@     W�@     ��@     ��@     ��@     �@     A�@     9�@    ���@    ��@     E�@     ��@    ���@    �t�@    ���@     %�@    �m�@    ���@    ��@    @��@    ��@     ��@    @��@     ��@    ���@     ��@    @v�@    @��@    `$�@    ��@    ���@    `�@     5�@    ���@    `��@    @��@    �W�@     k�@    ��@    ��@    �w�@     V�@    ` �@    @�@    �1�@     .�@    ��@     ��@    ���@    �]�@    @��@    @��@    ���@    `q�@    �r�@    @��@    ��@     I�@     ��@    ���@    ���@    �\�@    �>�@     �@    ��@     ��@     ��@    ���@    ��@    ���@      �@    ���@    ���@    �C�@     m�@     ��@     �@     Ч@     2�@      �@     D�@     ��@     �@     �q@     �^@      E@      @      �?        
�
predictions*�	   ���   ����?     ί@! `���N`@)��E�(&@2�	!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9�>�?�s���O�ʗ����h���`�8K�ߝ�E��a�W�>�ѩ�-�>��ڋ?�.�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�	              �?      �?      @      @      @      @      4@      3@      >@      2@      7@     �A@      <@      <@     �B@      ?@      6@      B@     �A@      C@      >@      ?@      ;@      7@      3@      9@      9@      6@      ,@      ;@      5@      3@      $@      .@      *@      &@      0@      $@      &@      @      @       @      @      @      $@      @      @       @      @      @       @       @      @      @       @       @      @       @      �?      �?              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @      @      �?              �?      �?      @      @      @      @      �?      @      $@       @      @      @       @       @      @      @      &@      1@      0@      .@      (@      2@      4@      0@      9@      5@      @@      B@      G@     �I@      F@     �J@     �K@     �Q@     �L@     �T@      T@     �W@      `@     �Z@      b@     `b@     @b@     �c@     @d@     @j@     �e@     @a@     @f@     �`@     @^@      Y@     @P@      J@      A@      0@      @      @      @      @        ��X�"1      ~���	8J�U���A*�b

mean squared error"Q@=

	r-squared �<
�L
states*�L	    0��   �!n@   �$[RA!^vu��Q��)%IKJ�A2�%�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�������:�%             @Q@     �v@     w@      �@     ��@     ��@     ؗ@     h�@     z�@     ��@     �@     `�@    �(�@     ��@     ��@    �"�@     q�@     ��@     ��@    �S�@     ��@    ���@    �Q�@    �+�@    ���@    �\�@    ���@    ���@    ���@    @��@    @��@    �n�@    �M�@    ���@    �0�@     ��@    `��@    @X�@     ��@     +�@    @��@    `��@    @�@    �B�@    `��@     ��@    �V�@    �j�@    @��@    `{�@    ���@    @�@     �@    `A�@    ���@    �g�@    �	�@    ��@     C�@     ��@     ��@    @��@    �:�@    @�@    �j�@    �j�@     O�@     �@    �Q�@    ��@    @{�@    �T�@    ���@     ��@     h�@     ��@     }�@     b�@    �	�@     ��@    ���@     y�@     |�@     ��@     ��@     ��@     �@     ��@     ]�@     ]�@     ϲ@     a�@     ��@     �@     ��@     .�@     ��@     D�@     �@     ��@     ڧ@     R�@     ,�@     ��@     l�@     4�@     �@      �@     ��@     ��@     ��@     �@     v�@     �@     x�@     ��@     \�@     ��@     T�@     �@     ��@     ��@     h�@     ��@     ̕@     h�@     (�@      �@      �@     �@     ܓ@     �@     @�@     ��@     (�@     �@     �@     t�@     �@     �@     ��@     Ў@     ��@     0�@     ��@     ��@     (�@     X�@     Ї@     p�@     �@     ��@     ȇ@     H�@     8�@     �@     �@     ��@     ��@     h�@     ��@     `�@     ��@     ��@     H�@     (�@     8�@     ��@     �~@      @     �@     �~@      {@     �{@     �y@     �z@     �y@     �x@      w@     `x@     �v@     pv@     `t@     `u@     �s@     �t@     �s@     Pt@     0r@     �s@     0q@     �p@     �q@     �o@     @p@      q@     `o@     �l@     0p@     �m@     �i@      k@     `g@     �i@     `i@      i@     �j@     `f@     �e@     `h@     @g@     �b@     @b@     @c@     @c@     `c@     �_@      `@     �`@     `a@      ]@     @^@     �]@     �Y@     �\@     �Y@      \@     @U@     @Q@     �W@     �W@     @V@     @T@     @P@     @X@     �U@     �Q@      T@     �R@     �L@     @U@     �S@      P@      Q@     �Q@      K@     �L@      K@     �H@     �F@     �L@     @P@     �O@      B@      H@      G@      :@      C@      >@     �C@     �A@      7@      C@      @@      ?@      5@     �@@      2@     �C@      ?@      7@      =@      1@      4@      5@      4@      5@      1@      9@      ,@      *@      3@      &@      0@      4@      ,@      3@      *@      ,@      ,@      *@      @      *@      @      $@      &@      $@      @      *@      $@      "@      @     @u@     H�@      0@      &@      &@      5@      1@      (@      $@      0@      2@      2@      0@      1@      3@      3@      4@      3@      0@      8@      =@      :@      6@     �A@      7@      D@      >@      B@      @@      >@      >@      <@      =@     �A@     �B@      E@      >@     �A@      D@     �F@     �F@      B@     �C@      D@      G@     �G@     �J@     �H@     �J@      E@      L@      O@     �M@      I@      F@     �Q@     �Q@     �I@      M@     @P@      S@      Q@     �S@     �S@     �R@     @Q@      S@     �W@     �S@     @Q@      Z@      T@     @S@     �V@     @Y@     �V@     @U@     @\@     @]@      ]@     �]@     �`@      _@      `@      ^@     @c@     @^@     ``@     �b@     �_@      c@      a@      c@      b@     @f@     �f@     �f@      i@     `g@     �i@     �d@      f@     @g@     �g@     `h@     @h@     `k@     `k@      n@     �k@     �l@     �q@     �m@     �o@      q@     �r@     0q@      p@     �q@     �v@     @s@     0s@     0t@      v@     �t@     Pu@     �x@     �w@     �u@     �y@     py@     0x@     �w@     Py@     �|@     @}@      z@     `z@     0{@     @�@     @�@     p@     �@     �@     ��@     p�@     x�@     �@     ��@     @�@     �@     �@     ��@     �@     ��@     �@     �@     ��@      �@     8�@     @�@     X�@     x�@     ��@     H�@     x�@     4�@     H�@     ��@     `�@     Ȑ@     p�@      �@     ̒@     `�@     ��@     �@     ܓ@     p�@     @�@     ��@      �@     ��@     t�@     Ԗ@     ��@     0�@     �@     �@     ��@      �@     $�@     F�@     .�@     ��@     ؠ@     $�@     H�@     "�@     �@     �@     ��@     ĥ@     :�@     ��@     ��@     ��@     �@     Ȭ@     ��@     Q�@     Ͱ@     X�@     6�@     ��@     �@     ��@     �@     ��@     ��@     �@     ��@     '�@     ��@     ��@    ���@    �=�@    �W�@    �8�@    �7�@    ���@    ���@     ��@     ��@    ��@    �W�@    ���@    ���@    @E�@    ���@     ��@     ��@     ��@     ��@    �N�@    `��@     �@    �o�@    ��@     �@    �e�@     T�@    � �@     �@    �3�@    �=�@    ���@    ���@    ��@    ���@    �2�@     C�@    ��@     ��@    `��@     )�@    ��@    `�@    ���@    �J�@    @<�@    @n�@    �P�@    ���@    �r�@     �@    @�@    �V�@    @��@    ���@     �@     5�@    ��@     ��@    ���@    ���@     ��@    ���@    ���@      �@     ��@    �6�@     �@     4�@     ��@     ��@     X�@     �@     ؀@     �@     `p@     �V@      @        
�
predictions*�	   ����    HO�?     ί@!  �ʹ$�) �Il�@2�
����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'�������6�]���1��a˲��*��ڽ�G&�$��R%������39W$:���['�?��>K+�E���>�f����>��(���>�ߊ4F��>})�l a�>pz�w�7�>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?ji6�9�?�S�F !?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?�������:�
               @      $@      6@      I@      O@      R@     �W@     @^@      _@     @_@      b@      `@     �b@     �[@     �\@     �X@     �X@      S@     �Y@     �M@     @Q@      M@      D@      E@      C@     �C@      C@      E@      A@      @@      >@      ;@      ,@      0@      8@      2@      &@      ,@      &@      @      $@      (@      @      "@      @      @      @      @      @      @      @      �?       @      @       @      @      @      �?       @       @      @      �?      @      �?       @       @       @      �?              �?              �?      �?               @      �?              �?              �?              �?              �?              �?      �?              �?      �?       @      �?      �?      �?      �?              �?               @      �?      �?      �?      �?      �?      �?      @      �?      @      @              @      @      @      &@      @      @      @      "@      "@      &@      "@      *@      0@      2@      (@      ,@      *@      .@      2@      8@      5@      .@      B@      8@      ;@     �A@      B@     �M@     �E@      E@     �B@      G@     �K@      O@     �K@      N@      O@      H@     �A@     �G@     �J@      J@      E@      F@     �I@      H@      P@     �D@      @@      7@      0@      @      $@      @       @        |n��1      ~���	��V���A*�a

mean squared error�'A=

	r-squared �w<
�L
states*�L	   ����   ��2@   �$[RA!n�z�y���)T�u���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             @`@     @x@     �z@     ��@     ��@     t�@     ؝@     p�@     v�@     �@     ��@     Թ@    ���@    �+�@     7�@     ��@    �!�@     ��@     ��@     �@    �A�@    �%�@    �L�@    @�@    @;�@    ���@    ���@    �W�@    ���@    �|�@    �w�@    `�@    ���@    @�@    @��@    �Y�@    ���@    ���@    ���@    ���@    `%�@    `�@    `��@     �@    ��@     �@    @��@     O�@     ��@    ���@    �R�@    @$�@    `k�@    ��@     ��@    �M�@    @��@     ��@     -�@    ���@     B�@    �f�@    �w�@    ���@     ��@    �&�@    ���@    �u�@    @�@    �:�@      �@    �!�@     ��@     ��@     ��@    ���@    ���@    �>�@     ��@     ��@     �@     ܺ@     �@     ^�@     �@     �@     �@     �@     ]�@     �@     ��@     ȯ@     ��@     ��@     @�@     J�@     ��@     �@     ڧ@     ئ@     6�@     ��@     ��@     H�@     ޣ@     £@     ��@     X�@     ��@     0�@     &�@     J�@     �@     @�@     ĝ@     (�@     ��@     p�@     l�@     P�@     �@      �@     �@     �@     �@     $�@     h�@     �@      �@      �@     (�@     T�@     <�@     ؐ@     p�@     �@     \�@     �@     �@     x�@     �@     ��@     ��@     ��@      �@     X�@     ȍ@     X�@     Њ@     ��@     (�@     �@     @�@     H�@     ��@     8�@      �@     Ȇ@     ��@     ��@     �@     ��@     ��@     0�@     `�@     ȃ@     ��@     �@     �@     0�@     ��@     ��@     �@     X�@     (�@     �@     }@     `~@     P}@     |@     �z@     Py@     `x@     �u@     `x@     �x@     `v@     �t@     �s@     u@     pu@     Pv@     0r@     �r@     �t@     �q@     Pr@     Pp@     �p@     �o@     �p@      q@     pq@     pp@     �o@     �j@      l@     `k@     `m@     �k@     `l@      l@     �g@     �f@     �e@     �h@      f@     �g@     `e@     �c@      b@     �e@      e@      e@      c@     �c@     �c@     �a@      a@     �_@     @_@      \@     @^@     �\@     �^@     @\@     �Y@     �X@     �\@     �Y@     �Y@     �V@     �W@      Y@     @S@     �T@     �R@     �T@     �M@      T@     �S@     �Q@      T@     �R@     @U@      O@     �L@     �O@     �J@     �P@      O@     �K@     �L@     �I@     �M@      H@      I@      H@      H@     �J@      F@     �C@      A@      @@      F@     �A@      F@      ;@      A@      >@      ?@      9@      ;@      <@      ;@      ;@      9@      ;@      6@      7@      7@      6@      9@      0@      7@     �@@      6@      5@      2@      *@      .@      0@     ȅ@     `�@      *@      9@      9@      9@      :@      7@      5@      <@      >@      1@      6@     �D@      8@      ?@      >@      <@     �E@     �I@     �B@      A@     �B@      ?@      G@      J@      C@     �C@     �E@      F@     �G@     �E@     �I@      K@     �H@     �F@      I@      I@      L@      N@      P@     �N@      P@     �M@     �N@      O@     �O@     �U@     @U@     @T@     �R@      R@     @Q@     @Q@      U@     �R@      R@      T@     �V@     @U@     �Y@     �X@      \@      Y@      X@      Y@     �[@     �^@      [@     @Z@     �Z@      `@      `@     �^@      `@     @]@     �`@     �^@      a@     @`@      c@     �a@     `c@     �c@     `d@     �b@     @c@     �g@     `d@     �g@      g@     �e@     �g@      h@     `n@     `k@     �h@      i@     �h@     `l@     @j@      j@     q@     �n@     �j@      p@     �r@      q@      q@     �t@     �p@     �p@     �r@     �q@     @r@     �t@     Pu@     �t@     �u@     �w@     �u@     �y@     py@      z@     `x@     py@     �z@      y@     @z@     0z@     pz@     �@      @     �~@     p�@     @     ��@     P�@     0�@     ȁ@     ��@     h�@     x�@     �@     �@     ��@     ��@     ��@     ؅@     ؆@     �@     8�@     �@     �@     �@     ��@     @�@      �@     @�@     p�@     x�@     ȍ@     �@     ؍@     Ў@     ��@     ��@     ��@     ��@     �@     ��@     ��@     D�@     l�@     T�@     ��@     �@     P�@     �@     �@     ��@     ؖ@     ��@     ��@     ��@     ܘ@     �@     h�@     ��@     L�@     X�@     ��@     ��@     6�@     �@     �@     ��@     ��@     �@     �@     z�@     @�@     "�@     V�@     ��@     ��@     ީ@     >�@     2�@     ��@     ¬@     ��@     ��@     ��@     ױ@     ��@     p�@     Q�@     ~�@     �@     8�@     #�@     <�@     v�@     �@     k�@    �{�@    ���@     r�@     ��@     )�@    ���@     Q�@     ��@     c�@     ��@    �d�@    ���@    �)�@    ���@     N�@    ���@    �?�@    @��@    @��@    �#�@    �y�@    ���@    �2�@    @��@    @N�@    ���@    @��@    �?�@    �'�@    `D�@    ���@    �R�@    ���@     {�@    ���@    @�@     ��@    ���@    @��@    �X�@    �"�@    @@�@    ���@     d�@    ���@     J�@     ��@    @��@    ���@     K�@    ��@    ���@    �R�@    ���@    @��@    �c�@     b�@    ���@    �L�@     ��@      �@    �|�@    ���@    ���@    �Q�@    ���@     ��@     Н@     ��@     ��@     �@     @�@     Ȃ@     H�@     �|@     Pv@      X@      �?        
�
predictions*�	   @L-��   @��?     ί@! @UN7@)F��2�Y�?2�
����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���>h�'��f�ʜ�7
������6�]�����Zr[v��I��P=��pz�w�7��})�l a���n����>�u`P+d�>I��P=�>��Zr[v�>��[�?1��a˲?����?f�ʜ�7
?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?�������:�
              �?               @       @      "@      (@     �D@     �D@      G@     �L@      R@     �T@     �L@     �U@     �T@     @T@     @R@      Q@     �P@      P@      I@      M@      K@     �J@     �B@     �D@      D@      B@      A@      :@      <@      8@      7@      8@      4@      1@      @      *@      $@      ,@      *@      @      "@      *@       @      @      @      @      �?      @       @      @       @      @      @      @              @      �?       @      �?      �?      @      �?              �?              @      �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?       @              �?       @              @              �?      �?      @              �?      �?      @      �?       @      @               @              @      @      @      @      @      @      "@      @      @      $@      @      "@      .@      "@      0@      0@      1@      4@      8@      4@      B@      =@      3@     �A@     �D@     �D@     �E@     �C@      J@      L@     �N@      K@     �P@     �V@     �S@     �Y@     @U@      Z@      X@      Z@     �\@     �Y@     �X@     �X@     �Y@     �V@     �Y@     �R@     �G@      J@      <@      "@      @      �?        0c�;21      "���	4�V���A*�b

mean squared errorY@=

	r-squared ��<
�L
states*�L	   @��   `��@   �$[RA!��J4���)y��]A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �h@     �x@      �@     Ԑ@     (�@     Ĝ@     P�@     ��@     ��@     ��@     ̫@     з@    @C�@     v�@    ��@     E�@    ���@     ��@     ��@     ��@    �R�@     ��@    �A�@    ���@    �f�@    ���@    �K�@    ���@     ��@     ��@    `��@    `&�@    ���@    ��@     w�@    ���@     ��@     0�@    `��@     ��@    ��@    ��@     ��@    ��@    �I�@    ��@    `��@    ���@    �D�@    �3�@    ���@    `�@    ���@    @��@    ��@     ��@    ���@    �J�@    �e�@    �Z�@    �N�@    @Y�@     
�@    ���@    ���@    @�@    @�@    �~�@     p�@    ���@    �"�@     �@    ���@     ��@     f�@    � �@     ��@     ��@     ƿ@     ��@     x�@     *�@     ��@     N�@     !�@     �@     ͳ@     ��@      �@     5�@     �@     �@     l�@     j�@     Ҫ@     ��@     ��@     x�@     ��@     ��@     X�@     V�@     \�@     �@     �@     �@     �@     �@     ڠ@     ��@     .�@     �@     ڠ@     ��@     ,�@     �@     ��@     ��@     ��@     ��@     `�@     (�@     �@     $�@     P�@     D�@     ��@     �@     ��@     <�@     ��@     ��@     �@     ��@     D�@      �@     ��@     ��@     ��@     Џ@     ��@     ��@     �@     ��@     ��@     ��@     Њ@     ��@     (�@     ��@     ؈@     X�@     Ћ@     ��@     ��@     `�@     0�@     �@     ��@     p�@     ��@     `�@     �@     �@      �@     `�@     ��@     0�@     �@      �@     ��@      �@     ��@     ��@     Ё@     X�@     P�@     `�@     �@     �|@     0{@     �|@     �z@     �{@     �y@     �{@     �w@     �x@     pw@     �w@     �v@     �u@     �v@     �v@     �v@     0t@     �u@     0t@     �r@     �s@     `s@     �q@     �r@     Pq@     �q@     �p@     pq@     �p@     �o@     p@     �l@     �l@     �l@     �n@     @n@     �i@     �l@     �k@     �h@     �j@     �g@      k@     �f@      h@     �g@     �e@     `e@      g@     @d@     @d@     �c@     �c@      c@     �d@     �d@     �b@     �a@     �`@      `@      a@      \@     @\@     @[@     @^@      \@     �X@     @`@     @T@      Z@     �\@     �W@     �T@     �Z@     �W@     @Y@     @W@     �T@     �P@     �P@     �P@     �R@     �O@     @Q@      Q@      Q@     �R@     �T@      N@     @P@      P@      J@     �N@     �P@     �G@     �K@     �J@      K@     �J@     �F@      K@      K@      A@      K@     �H@     �E@     �C@     �K@      <@      E@      8@      <@      9@      @@      A@      .@      @@      =@      :@      <@     �B@      ;@      3@     ��@     ��@      A@      <@      =@      A@      >@      C@     �D@      B@      A@      D@     �E@      ?@      E@      F@     �H@     �F@     �K@      I@      H@     �G@     �K@     �I@     �D@     �L@      H@     @Q@      N@     @P@     �I@     �Q@     �P@     �G@     �J@      F@     �O@      V@     @R@     �P@      S@     @S@     �U@     �S@     @S@     �U@     �U@      W@     @X@      Z@      S@     @Y@     @X@      Y@     �Y@     �a@      Y@     @W@      \@     �`@     �`@     �]@      a@      `@     `a@     �]@      c@     @a@     @`@     `b@     �_@     �d@     �c@      b@     `b@     �d@      g@      d@      c@     `d@     �d@     `h@     �m@     �i@     �h@     �i@      j@      i@      i@     �k@     @j@      k@     �j@      l@     0p@     @m@      q@     pr@     Pp@     ps@     @q@     �r@     �r@     �r@     �s@     �t@     �s@     pu@     w@     �s@     �t@      x@      z@     �w@     �w@     @x@      z@     0x@     py@     �{@     �z@     Px@     p{@     p~@     @~@     �@     �}@     8�@     ��@     ��@     p�@     ��@     h�@     `�@     �@     ��@     ��@     ��@     ��@     Ђ@     8�@     ��@     @�@     H�@     0�@     0�@     ��@     ��@     x�@     ��@     h�@     H�@     @�@     8�@     x�@      �@     X�@      �@     p�@     H�@     ��@     p�@     (�@     �@     ��@     8�@     l�@     ̓@     ��@     Ȓ@     ��@     �@     L�@     Ĕ@     ��@     h�@     �@     l�@     X�@     P�@     �@     `�@     ��@     |�@     �@     �@     $�@     ܛ@     �@     Н@     \�@     T�@     ��@     ��@     �@     ��@     8�@     ,�@     $�@     ޣ@     ��@     h�@     J�@     "�@     v�@     `�@      �@     ��@     �@     �@     j�@     �@     ��@     E�@     +�@     �@     Ų@     Ĵ@     �@     ȶ@     �@     a�@     ��@     e�@     ;�@     ��@     ��@    � �@     ��@     `�@    ���@    �a�@     ��@    ���@    ���@    �l�@     ��@    �?�@    @�@    ���@    ���@    �J�@    @��@    @X�@    ��@     ,�@    �A�@    ���@    ���@    `�@    @~�@    @��@    �
�@    @��@     2�@     /�@    ���@    ���@    �t�@     ��@    ���@    `��@    @>�@    ��@     O�@    �I�@    `H�@     h�@    ���@    ���@     Q�@    `��@    @t�@     ��@    ���@    ���@    �f�@    ���@    @x�@     ��@    �{�@    ���@     ��@     s�@     �@     �@     ��@    ���@    ���@     ��@    �1�@    �~�@     ��@     h�@     �@     �@     ��@     P�@     ��@     �@     �}@     �}@     �o@      @        
�
predictions*�	    '2��   �S��?     ί@! �/�$� �)==Q��	@2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��T7����5�i}1�>h�'��f�ʜ�7
�1��a˲���[����Zr[v��I��P=��pz�w�7����(��澢f���侄iD*L�پ�_�T�l׾K+�E���>jqs&\��>���%�>�uE����>I��P=�>��Zr[v�>�FF�G ?��[�?1��a˲?�5�i}1?�T7��?��ڋ?�.�?U�4@@�$?+A�F�&?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?�������:�
              �?       @      @      ,@      9@     �E@     �N@     �Y@     �V@     �W@     �Y@      W@     �[@     �U@     �X@      T@     �U@     �R@      S@     �Q@     �S@     �J@      G@      G@      E@     �I@      A@      :@     �@@      8@      :@      ?@      <@      3@      1@      &@      0@      &@       @      *@      @      @      @      @      @      (@      @      @      @      @       @      @      @      �?      @      @              @      @       @      �?      �?               @       @               @      �?              @      @      �?              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?               @               @              �?      �?      �?       @      @      @               @       @      @      @      @       @      @      @      @      @      @      *@      "@       @      @      $@      ,@       @      1@      2@      2@      0@      8@      0@      3@      7@      5@      <@      C@      F@      C@      I@     �K@      H@     �L@     �Q@      P@     @Q@     @Y@      V@     @W@     �]@     �[@     �\@      Z@     �Z@     @W@     �T@      F@     �D@      9@       @      &@       @      @       @      �?        �x9�"1      ~���	�k�V���A*�b

mean squared error�#?=

	r-squared�8�<
�L
states*�L	   `@��   ���@   �$[RA!e�	z���)��=���A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �n@     ؃@     @�@     `�@     \�@     t�@     �@     ��@     ¡@     ڧ@     \�@     �@     �@    �]�@    �_�@    �!�@     j�@    ���@    ���@     ��@     7�@    �p�@     (�@    @J�@    �Z�@    ���@    �9�@    @��@     (�@     ��@    `�@    `��@    ���@    `�@    @��@     �@    ���@    @��@    �c�@     ��@    �c�@    @��@     ��@     ��@     ��@    @��@    ���@    ���@    @��@    ��@    ���@    `��@     ��@    @��@     i�@     ��@     ��@    ���@    @O�@    �7�@    ���@     �@    �@�@    @��@     �@    @��@    @��@    �>�@     :�@     \�@     ��@     \�@    ���@     p�@     ��@    ���@     Z�@    ���@     Ǿ@     ��@     Ի@     B�@     ��@     ��@     ��@     �@     c�@     6�@     ��@     ��@     �@     F�@     2�@     "�@     t�@     <�@     ��@     �@     ��@     <�@     D�@      �@     ޣ@     �@     Ȣ@     ��@     �@     ��@     D�@     �@     �@     P�@     j�@     t�@     x�@     ��@     ��@     ��@     $�@     Ԛ@     �@     ș@     �@     `�@     ��@     ��@     <�@     �@     ��@     d�@     ��@      �@     ��@     x�@     �@     �@     d�@     \�@     �@     ��@     ��@     ��@     @�@     H�@     0�@      �@     ��@     ��@     ��@      �@     ��@     ��@     X�@     ��@     ��@     @�@     ��@      �@     ��@     ��@     ؆@     @�@      �@     Є@     �@     �@     8�@     ��@     �@     ��@     x�@     ��@     ��@     ��@     Ѐ@     ��@     �@     0�@     ��@     �@     �}@     ��@     �@      }@     �}@     �|@     �}@     �z@     @{@     @z@     pw@     px@      x@     �x@     {@     �y@     �u@     �v@     �u@     �w@     pu@      t@     �u@     Pv@     �r@     �q@     `q@     Pr@     Ps@     �q@     �r@     �o@     @q@     @l@     p@      o@     �o@     �l@     �n@     �l@     �h@     @j@     �j@      l@     �k@     �f@     @h@      h@     �f@     �g@     @g@     �f@     �f@     �c@     �g@     �f@     �b@     @c@      b@     `b@     �`@     ``@     �b@     �`@     �`@     �_@      ]@     �^@     @[@      a@     �_@     �Z@      `@     �]@      W@     @Z@     �V@     @Y@     �W@     �X@     �X@      U@     �W@     �T@      Z@     �X@     �U@     �T@      U@     �S@      O@     �V@     �Q@      M@      R@      K@     �M@     �N@     �M@      L@     �P@     �H@     �H@     �J@     �N@     �L@     �F@      G@     �K@      :@      A@      E@      >@      C@     �A@      C@      B@      C@      C@      C@     �A@     �A@     `�@     d�@     �E@     �@@      I@      C@      G@     �D@     �E@     �F@      D@      F@     �G@     �P@      K@     �O@     �P@     @Q@      P@      J@      K@      N@     @P@     @R@      Q@     �P@      O@     @S@     @T@     �R@     @X@     �S@     �S@     �S@     �T@      U@     @W@     @X@     �V@     @U@      U@     @U@     �\@     �X@     �Z@     �Z@     �W@      Y@      ]@      \@     @]@     �_@      [@     `b@     ``@      \@     �[@     �`@     �[@     `d@      `@     �b@     �b@     �`@     �c@     �d@     `c@      f@     @e@     �d@     �d@     @f@      g@     �h@      e@      k@     �h@      j@     �j@     @o@     `l@     �l@     �j@     @n@      p@     `o@      l@      q@      p@     `r@     �q@     r@     �q@     pu@     �t@     Pq@     �q@     `s@     Ps@     0v@     `y@     �w@     �u@     0w@     �y@     �w@      w@     �x@     �v@     �z@     @x@     �x@     �z@     Pz@     �y@     �~@      ~@     �~@     �|@      �@     Ȁ@     �@     ��@     ��@     h�@     8�@     ��@     X�@     p�@     ��@     x�@     H�@     �@     H�@     ��@     ��@      �@     @�@     p�@     0�@     ��@     `�@     h�@     ��@     Ј@     ��@     ��@     �@     (�@     ��@      �@     x�@     �@     h�@     0�@     �@     (�@     D�@     |�@     ̑@     X�@     P�@     <�@     �@     ��@     ��@     �@     ��@     (�@     p�@      �@     l�@     �@     $�@     �@     \�@     �@     L�@     ��@     �@     �@     �@     ��@     0�@     T�@     ��@     D�@     ��@     ĝ@     ��@     N�@     t�@     ��@     �@     �@     r�@     ��@     �@     ��@     ��@     ��@     ��@     z�@     ��@     Ȩ@     ��@     
�@     �@     �@     �@     V�@     ��@     r�@     ��@     ��@     �@     ��@     �@     }�@     ��@     ��@     ��@     �@     �@     ��@     ��@     ��@    �n�@     ��@    ���@    �x�@     b�@    �_�@    ���@     X�@    ��@    �V�@    @��@    @��@    �:�@    @��@    @A�@    �R�@    �5�@     ��@     ��@    ���@     ��@    `��@    �]�@    ��@     ��@    ���@    �8�@    `8�@    `/�@    ���@    ���@    ��@    ���@    ���@     /�@    �<�@    ���@    �z�@    ���@     
�@    �?�@     ;�@    ��@     ��@    @��@     E�@    ���@     /�@    ���@    �i�@    �`�@    ���@     n�@    �0�@     d�@    @��@     5�@     `�@    ���@    �1�@    ���@    @��@    �o�@     �@    �[�@     �@     ��@     P�@     ��@     0�@     Ȁ@     P}@      x@     �x@     �t@     pr@      D@        
�
predictions*�	   ���ÿ   ����?     ί@! �l� �6�)�ٓ�/�@2�
yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��vV�R9��T7����5�i}1���d�r�x?�x����Zr[v��I��P=��pz�w�7��})�l a�8K�ߝ�a�Ϭ(���Zr[v�>O�ʗ��>��d�r?�5�i}1?�.�?ji6�9�?�[^:��"?U�4@@�$?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�������:�
              �?      �?      @      @      (@      2@      ;@     �L@      T@      `@      b@     @b@     �]@     �[@      Z@     @^@     �V@     �T@     �R@     �T@     �Q@     @P@     �K@      H@      E@     �C@      I@     �C@     �A@      C@     �B@     �C@      <@      9@      .@      *@      *@      ,@      0@      .@      3@      @       @       @      &@      "@      @       @      @      @      @      @      @       @       @      @       @       @       @       @       @      @      �?      �?              �?      �?      �?               @              �?      �?              �?       @               @              �?              �?              �?               @               @              �?              �?              �?      @       @      �?       @      �?      @      @       @       @      �?       @      @      @       @      @      @       @       @       @      @      @      (@      &@      @       @      *@       @      *@      &@      (@      .@      0@      6@      A@      4@      >@      >@      ;@      >@      F@      D@      F@      B@     �F@     �J@     �P@     �R@     @T@     @T@     @X@     �Y@     �U@      Y@     �U@     @Q@     �Q@     �S@     �G@      H@      @@      1@      &@      *@      @      @      @       @      �?              �?        � Ѧ1      "���	іfW���A*�b

mean squared error==

	r-squared��=
�L
states*�L	    ���    ��@   �$[RA!���҆��)���0�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              m@     <�@      �@     �@     8�@     L�@     p�@     "�@     :�@     @�@     |�@     y�@    �"�@    �&�@     ��@    �T�@    �=�@     ��@    ��@    �0�@    ���@     &�@    @g�@    @��@     u�@    �h�@     ��@    �i�@    ��@    �X�@     ��@    `��@    ��@    `"�@    �$�@    ���@     ��@     \�@    ��@     ��@    @��@    �4�@    ���@    ���@     |�@    � �@    ���@    �3�@     ��@    �W�@    ���@     ��@     ��@    �{�@    ��@    ���@     ��@    @��@    �r�@     ��@    @��@     ��@    ���@    �Y�@     ��@    �!�@    @x�@    �8�@    ���@    ��@    ���@     ��@    ���@    �K�@     ��@     ��@    �z�@     -�@    �W�@     �@     ��@     غ@     ��@     ��@     Ķ@     µ@     X�@     ��@     ��@     Ȱ@     ��@     
�@     `�@     b�@     ��@     B�@     
�@     ��@     :�@     �@     >�@     ��@     �@     ��@     �@     d�@     ��@     r�@     4�@     �@     "�@     4�@     (�@     ��@     D�@     H�@      �@     d�@     l�@     ؙ@     ��@     T�@     �@     ��@     ��@     ��@     �@     ��@     |�@     ��@     �@     P�@     (�@     X�@     ��@     đ@     \�@     Ȑ@     ��@     4�@     Ȏ@     �@     (�@     ��@     @�@     ،@      �@     ��@     ��@     (�@     Њ@     Њ@     ��@      �@     H�@     ��@     ��@     ��@     0�@     �@     H�@     Ȅ@     (�@      �@     ��@     p�@     H�@     �@     ��@     ؂@     �@     8�@     ��@     ��@     ��@     P�@      @     `�@     �@     0@     �}@      ~@     �}@     0|@     �@     `~@     �y@     �y@     `z@      y@     �y@     �w@     0w@     �x@     �w@     0v@     v@      v@     @u@     0s@     pt@     �s@     �u@     �u@     �s@     �u@     �r@     �p@     ps@     �q@     0q@     Pr@     �o@     @o@     �o@     �m@     `j@     �l@     �m@     �o@     �j@     �n@      g@     @m@     �j@     �j@     �j@      e@     `f@     �d@      f@     �g@     �c@     �g@     �e@      e@      f@     `c@     �c@     @c@     �a@     �d@     @`@     @c@     �`@     @_@     �c@      `@     �]@     �`@     �^@      X@      ]@     @Z@     @Y@     �W@     @_@     �Y@     �Y@      W@     �X@      X@     �V@     �T@      V@      Q@      N@      U@     �V@     @U@      T@      R@      U@     @R@     �N@     @R@      P@      I@     �P@      Q@     �M@     �N@     �L@      O@      J@     @Q@     �E@     �I@      B@      E@      F@      @@     �G@      C@      B@      E@     �G@      A@      A@      C@     �D@      =@     t�@     �@     �J@     �F@      N@      O@      N@     �P@      L@     �P@     @Q@      O@      L@      S@      M@      L@      F@      K@     �Q@     @R@     @Q@     �Q@     �S@     �P@     �Q@      V@     �S@     �U@      T@     �U@     @V@     �S@     �V@      X@      W@     �Y@      X@     �[@     @X@      \@     �V@     �X@     @]@     �V@     �^@      X@     @^@      _@     �[@     @Z@     �`@     �`@      `@      ^@     �a@     �a@      `@     @e@     `d@     �c@      e@     @d@      e@     �d@      h@     �e@      g@      g@     �g@     �e@     �j@     @j@     �g@     `g@     �g@     �l@     s@     �j@     @m@      m@     �o@     Pq@     p@     �m@     �m@      o@     �p@     Pp@      n@     �q@     @t@     �r@     s@     �q@     0w@     0u@     0s@     �t@     �v@     �t@     �t@      v@      x@     t@     �x@     @w@     �x@     0x@     �w@     �{@     Pz@     �|@     �|@     �z@     0{@     �|@     �{@     �~@     8�@     P~@     H�@     @@     �@     ��@     0�@     ��@     ��@     ��@     ��@     X�@     �@     �@     �@     �@     (�@     x�@     8�@     ��@     0�@     Ȇ@     �@     0�@     ��@     �@     �@     P�@     ��@     (�@     p�@     ȋ@     H�@     p�@     �@     ،@     �@     H�@     X�@     l�@     ��@     ��@     X�@     ̐@     ��@     ��@     $�@     ��@     ��@     ,�@     H�@     �@     H�@     �@     ��@     H�@     ��@     |�@     ,�@     `�@     ��@     �@      �@     ��@     ��@     �@     �@     x�@     ��@     �@     &�@     l�@     |�@     \�@     ��@     ڠ@     t�@     ��@     L�@     p�@     �@     ޤ@     ��@     b�@     ��@     ��@     N�@     &�@     �@     b�@     ~�@     �@     ��@     ��@     M�@     ��@     o�@     e�@     ��@     U�@     2�@     ��@     �@     �@     ��@     �@     �@     ��@     ��@     ��@     �@     ��@    �`�@    �e�@     r�@    ���@     ��@    @��@    ���@    @��@    ���@    ���@    @��@    @8�@    @(�@    �2�@    @}�@     ��@    ���@     ��@     4�@    �*�@    ���@     ��@     ?�@    �Q�@    ���@    `��@     t�@    ���@    �/�@     a�@     ��@    `�@    �8�@    ���@     ��@    @��@    �b�@    �\�@    �D�@    �;�@    �S�@     �@    ���@     ��@    � �@     ��@    ���@    ���@    ���@    �Y�@    ��@    @$�@    @��@     =�@    ���@    ���@    �$�@    �r�@    ��@     v�@    ���@     ;�@     ��@     �@     ��@     �@     ��@     ��@      |@     `v@     `v@     Ps@     �q@     �V@        
�
predictions*�	   �J���   �h�?     ί@!  ֬�=2@)�[�4�9@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$���d�r�x?�x��>h�'��f�ʜ�7
�pz�w�7��})�l a���Zr[v�>O�ʗ��>f�ʜ�7
?>h�'�?x?�x�?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�������:�
              @      @      @      @      $@      (@      7@      B@     �Q@     �W@     �_@     ``@     �_@     ``@     �`@     �`@     �V@     �T@     @X@      R@     �P@      O@     �I@     �N@     �M@      E@      B@      >@      6@     �@@      9@      :@      6@      <@      8@      $@       @      $@      @      (@      &@      $@      $@      @      "@       @      @      @      @      @      @       @               @      �?       @      @      @      �?              �?              @      �?              �?              �?              �?              �?              �?              �?      �?              �?      �?      �?       @              �?       @       @              �?              �?      �?              �?      �?       @               @       @      @      @      �?      @      (@      @       @      @      �?      @      ,@      &@      "@      "@      (@      "@      .@      *@      6@      3@      5@      =@      7@      B@      >@      :@      B@     �B@      B@      B@     �E@     �F@      K@      P@      K@     �L@      O@     �K@     �T@     �M@     @P@     �M@     �Q@     �P@     �T@     �R@     �M@      K@      O@     �F@      D@      C@      A@      6@      7@      $@      2@      (@      @       @      �?       @      �?               @        ���b0      y;l�	��W���A	*�`

mean squared error��<=

	r-squaredP=
�L
states*�L	   @���   ��@   �$[RA!�T$l���)E@�fA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&              v@     ��@     X�@     \�@     �@     ��@     Л@     (�@      �@     ��@     ®@     �@    �<�@    �/�@    �B�@    �4�@    �/�@    ��@     l�@     ��@     0�@    �S�@     ��@     X�@    @b�@     �@    @y�@     K�@     �@     G�@    `o�@    ��@    ���@    �X�@    ���@     �@    ���@    ��@    �-�@    �_�@    ���@    ��@    �K�@    �&�@    @��@    �
�@    @��@    @a�@    ��@     ��@    ��@     R�@    �l�@     ��@    @��@    @��@     ��@     ��@    ���@    �*�@    @��@     f�@    ���@    ���@     �@    @��@    @;�@    ���@     F�@    ���@    ���@     ��@    ���@     ��@     j�@    ���@     ��@     ��@    �:�@     �@     �@      �@     ӹ@     ��@     ��@     4�@     ��@     �@     �@     Q�@     а@     ��@     �@     ~�@     �@     J�@     ��@     v�@     j�@     ��@     ��@     B�@     ̣@     f�@     ��@     6�@     ��@     �@     J�@     H�@     ��@     �@     |�@     �@     ��@     ��@     h�@     0�@     x�@     �@     <�@     ��@     ��@     4�@     ��@     ��@     ��@     �@     D�@     ��@     h�@     ��@     ��@     �@     t�@     H�@     �@     ��@     d�@     ��@     ��@     �@     �@     �@      �@     ؍@     Ѝ@     ��@     �@     ��@     `�@     �@     �@     0�@     ؉@     p�@     ��@     X�@     ��@     p�@     @�@     �@     @�@     ��@     ��@     �@     x�@     P�@     �@     ��@     ��@     ȃ@      �@     ��@     ��@     `�@     ��@     ��@     �@     `@     �@     �~@     x�@     �~@     |@     �@     �{@     �~@      {@     �y@     |@     �{@     |@     �x@     `x@      w@     �w@      x@     x@      w@     0w@      u@     �u@     0s@     �t@     0s@      r@     s@     �q@     �s@     �r@     �r@     Pr@     pr@     �q@      q@     0p@     pq@      n@      q@      o@     `n@     `l@     �j@      k@     �i@     �k@     �h@      k@      h@     `g@     �e@     @i@     �e@     �f@      e@      f@     �d@      e@      g@      h@     �b@     �a@     �b@     `b@     �d@     �b@      ^@     �a@     �]@     �^@     @\@      a@     �_@     @\@     �]@      _@     �^@     @X@     �T@      [@     @X@      Z@      W@     �W@      U@     �R@     �X@     �W@      P@     @S@      R@     @V@      M@     �R@      T@     �Q@      W@     �K@     �N@      Q@     �J@      T@      I@     �N@      Q@     @Q@      J@     �I@      L@      I@      J@      E@      H@      K@     �G@      J@     �B@     �B@     �H@     �H@      :@     ��@     4�@     �N@      N@      O@      O@      Q@     �J@     �M@      P@     �Q@     �P@     @R@     @R@     @T@     @R@     @U@     �P@     @T@     �U@     �S@     �T@     @U@     �S@     @V@     @V@     �X@     @X@     �]@     @\@      Y@      Z@     �]@     �[@     @^@     �[@     �Z@     @X@     �X@     @]@     @_@     �`@     �_@     �`@     �]@     ``@     �[@      b@     �b@     @a@     �c@     �d@     @c@     �e@     @c@      b@     �e@     �g@      b@     �f@      d@     �e@     �h@     �f@     �h@      j@     `g@     �j@     �j@     `j@     �m@     �l@     �l@     �m@     �q@     Pp@      o@     0s@      p@     �o@      s@     `o@     �p@     `p@     `o@      r@     �q@     �r@     �o@     `t@     �v@     �s@      u@     �u@     @w@     0v@      v@     x@     �x@     0u@     �v@      w@     �w@     �z@     �y@     �z@      {@     �y@      z@     0{@     `}@     �}@     �@     `|@     `}@     �|@     �}@     �}@     �@     �~@     0~@     ��@     ��@     ��@      �@     p�@     x�@      �@     ��@     x�@     X�@     p�@     ��@     x�@     ��@     �@     ��@     (�@     �@     0�@     �@     �@     ȉ@     8�@     P�@     ��@     �@     ��@     ��@     ��@     �@     ��@     @�@     @�@     Ȏ@     ��@     Ѝ@     Ў@     ȏ@     l�@     ,�@     ܐ@     �@     �@     ��@     $�@     �@     \�@     ��@     ԓ@     ��@     8�@     �@     �@     @�@     ��@     \�@     �@     ��@     `�@     ��@     L�@     t�@     �@     ��@     `�@     0�@     ��@     |�@     О@     �@     �@     ��@     �@     �@     8�@     �@     ��@     Z�@     ̥@     ��@     2�@     B�@     f�@     ��@     ��@     ��@     «@     ��@     �@     ®@     �@     L�@     ӱ@     d�@     �@     K�@     d�@     ��@     �@     �@     a�@     Ѽ@     ��@     ��@     K�@     o�@    ���@     �@    ���@    �W�@     ��@     ��@    �i�@    ��@    @D�@    ���@    @��@    @2�@    ���@    �b�@    �u�@    �(�@    @��@    �f�@    `1�@    �a�@    �`�@     y�@    �^�@     |�@    ���@    @L�@    ���@    �q�@    �@�@    ���@    ��@    ���@    ���@    �W�@     �@    �Q�@    ��@    ���@    �c�@    @��@    ��@    �[�@    �.�@    �w�@    �:�@    �:�@     >�@     ��@    �L�@    ���@    ���@    @-�@    @��@    �Q�@    @g�@    @��@    ��@     ��@    �V�@    �R�@    ���@     ��@     �@    �c�@    @f�@     �@     ��@     T�@     H�@     h�@     ��@     �|@     �v@     �s@     �i@     �p@      `@        
�
predictions*�	   ��5��   ����?     ί@!  �bCS1�)��T�'@2�	��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��[^:��"��S�F !�ji6�9���.������?f�ʜ�7
?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�������:�	              @      "@     �I@     �U@     �]@     `a@     @g@     @h@     `f@      g@     �f@     @d@      a@     �]@     �\@     @X@      Y@      R@     @R@     @P@      L@      H@      D@      D@     �A@      ?@      <@      ?@      <@      ,@      5@      (@      $@      *@      "@      @      $@      &@      "@       @      (@      @      @      @      @      @      @       @      @      @      �?      @      �?       @              �?       @               @              @       @               @      �?      �?              �?              �?      �?              �?              �?               @      �?      �?      �?              @      @      @      @      @      @      @       @      @      "@      @      @      @      $@      "@      "@      "@      $@      (@      "@      1@      (@      2@      0@      *@      5@      3@      :@      :@     �@@      9@      6@      <@      :@      <@      @@      9@      E@     �A@      C@      >@      =@     �F@      C@      E@      A@      =@      B@      B@     �D@      <@      ?@     �@@      5@      :@      4@      *@      *@      (@      (@      ,@      ,@      "@      "@      @              @      �?       @              �?        ��ٟ�1      }�j�	��IX���A
*�b

mean squared error8�;=

	r-squared��0=
�L
states*�L	   `"��    S�@   �$[RA!XiO���)���;A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             X�@     Ж@     ؐ@     �@     �@     8�@     `�@     x�@     6�@     ��@     �@     p�@     ��@     !�@     �@    @��@    �_�@    �p�@     }�@     ��@     ��@     �@    �r�@     ��@     ��@    �8�@    ���@    �Y�@    ��@    �A�@    @Z�@    �F�@    `��@    �f�@    `�@    ���@    �N�@    ���@    �n�@     ��@     :�@    �F�@    ���@    ���@     y�@    �D�@    ���@    ���@     Z�@     ��@    �o�@    @��@    �F�@    �I�@    ���@    `��@    ���@    ���@    ���@    ���@    ���@    @��@    @
�@    �^�@    @��@    ���@    @��@    �C�@    ��@     i�@    �Q�@     W�@     ��@    �u�@     ��@     ��@     2�@     ��@     �@     B�@     �@     ܻ@     ��@     w�@     �@     ��@     [�@     �@     �@     ͱ@     ��@     �@     F�@     (�@     ��@     ��@     `�@     Ш@     \�@     �@     ��@     ��@     (�@     ��@     �@     �@     ҡ@     >�@     ��@     �@     >�@     ��@     ܟ@     ĝ@     �@     ��@     @�@     ��@     p�@     �@     8�@     ��@     <�@     ��@     x�@     �@     �@     ��@     D�@     ��@     ̓@     D�@     ��@     ��@     ��@     ؑ@     �@     |�@     H�@     4�@     x�@     ��@     ��@     t�@     ��@     p�@     ��@     ��@     H�@     ��@     �@     ��@     H�@     ��@     �@     ��@     �@     ��@     ��@     H�@     �@     p�@     �@     �@     �@     ��@     ��@     ��@     �@     ��@     ȃ@     ��@     0�@     H�@     ��@     P�@     ��@      �@     ��@     0�@      �@     P�@      �@     �@     0~@     �@      |@     @}@     0|@     |@     �}@     `{@     0z@     �x@     @x@     0z@     �w@     Pw@     �x@     �v@     �y@     �x@     �v@     �v@      x@     �u@     �w@     0s@     �r@     �t@     �q@     0q@     Pq@     �q@     �s@     pr@     @r@      q@     Pp@     `q@     �q@     �m@      k@     @p@     �q@      p@      m@      m@     �k@     `m@     �g@     �k@     `g@     �j@     �k@     �f@     �i@     �g@     �e@     �g@     `f@     �d@      h@     `b@     @g@     �d@     �c@     �c@     `d@     �_@      b@      a@     �a@     @c@     �b@     �_@      b@      _@     �\@      `@     �^@     �[@     @^@     @_@     �Z@     �\@     �Z@     �W@     �[@     �W@     �V@     �W@     �V@     �P@     �U@     @S@     @W@     �V@     @S@      P@      X@     �R@     �S@     @R@      T@     @U@      Q@      N@     �O@      L@      O@     �G@     �P@     �P@      L@     �N@      K@     �I@     �M@      I@     �H@     �H@     �@     Ҧ@     �P@     @T@      R@     �P@     �T@     �O@     �O@     �Q@     �T@     @T@     �R@     �M@     @V@     �X@     @X@      [@      Z@     �W@     �V@     �S@     �V@      ]@     @Z@     �V@      ]@     �Y@      V@      `@     @Z@     �\@     @`@     �Z@     ``@     @\@     �\@      `@      a@     �[@     �_@     @`@      b@     @c@     @d@     �d@     �b@     `b@     �b@     �a@     �d@     `c@      g@      f@     �g@     �e@     �h@     @f@     �h@     @j@     @i@     �l@     �h@     `i@      j@     `i@     �p@     �n@      n@     �o@     @p@     `p@     @o@      o@     @p@     `n@     p@      q@     `q@     �p@     Pr@     �q@     pu@     �q@      r@     t@     �s@      w@     x@     �v@     �u@     �u@      u@     �v@     v@     �v@     `u@      w@     @v@     �x@     py@     �w@     �z@     �{@     �y@     z@     �|@     �{@      |@     �x@     �{@     ~@     �}@     P�@     �@     �}@     �~@     ��@     �@      �@     Ȁ@     ��@     h�@     �@     ��@     ��@     H�@     (�@     ��@     ��@     �@     ��@     X�@     ��@     `�@     Ȇ@     ��@     І@     ��@     �@     h�@     ��@     �@     ��@      �@     8�@     �@     `�@     p�@     8�@     ��@     `�@     Ѝ@     ��@     ��@     �@     (�@     ��@     ,�@     ��@     ��@     đ@     X�@     �@     D�@     ��@     ��@     ԓ@     ��@     ��@     |�@     4�@     ��@     ��@     ��@     x�@     �@     ȗ@     0�@     ș@     ��@     |�@     @�@     �@     ԝ@     0�@     8�@     ��@     \�@     ��@     ڠ@     �@     ��@     �@     Σ@     ��@     .�@     ��@     ��@     @�@     �@     ��@     Ч@     �@     �@     ��@     N�@     *�@     Z�@     �@     
�@     ��@     �@     4�@     ��@     ��@     Ǵ@     /�@     Ƿ@     ��@     ��@     h�@     8�@     �@     ��@     ��@     ��@    ��@     ��@    ���@    ���@     A�@     T�@    ���@    �t�@    ���@    @�@    �f�@    @��@    @��@    �N�@    �>�@    @��@    @��@     ��@    @��@    ���@    `��@    @y�@    �s�@     W�@     ��@    `��@    ���@    ���@    �y�@     �@    ���@    �'�@    �f�@    ���@    @��@    ���@     ��@     [�@    ��@     6�@     ��@     ��@     C�@     r�@    `��@    �p�@    �5�@     ��@    �F�@    @[�@    �o�@     ��@     \�@    @��@    �#�@    @W�@     5�@    ���@     ��@     ~�@     "�@     [�@     ��@     ��@    �:�@     ��@     ��@     �@     ��@     P�@     Є@     ��@     �w@     `m@      e@     �n@      f@        
�
predictions*�	    �1��   `w<�?     ί@!  4���@)aS1�y�!@2�
��]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7���x?�x��>h�'��>�?�s���O�ʗ����u`P+d����n�������(���>a�Ϭ(�>1��a˲?6�]��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?�������:�
              �?       @      6@      8@     �H@      \@      `@     �d@     �f@     �f@     �i@      d@      a@     �_@     �\@     @U@     �X@     �U@     �M@     �P@      I@     �K@      J@     �B@      C@      :@      7@      5@      3@      6@      7@      8@      5@      .@      .@      @      *@      (@      (@      "@      @       @      @      @       @      @      @      @       @      @       @      @      �?       @      �?              @       @      �?      �?       @      �?       @      �?              �?               @              �?               @              �?              �?              �?              �?              �?      �?      �?              �?      �?       @      �?              �?       @              �?      �?              �?       @      @              @      @      @              @      @      @       @      @       @       @      "@       @      (@      (@      &@      ,@      *@      (@      0@      ,@      3@      6@      9@      1@      D@      >@      ?@      A@     �B@      L@     �H@      B@      I@      D@     �I@      I@      B@     �H@      B@      ;@     �I@     �D@      E@     �D@      G@     �B@      A@      <@      7@      9@      3@      <@      5@     �A@      ,@      $@      .@       @      &@      "@       @      @      @      @       @              @              �?        7�ޢ�1      |�4�	���X���A*�c

mean squared error�6=

	r-squared�E�=
�L
states*�L	   `���   ���@   �$[RA!��2����)o�y��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     �@     l�@     ��@     �@     L�@     ؘ@     8�@     L�@     ��@     �@     ٺ@    @Y�@     ��@     ��@    �9�@    ��@    ���@     �@    ���@     ��@    @�@    �N�@    ���@    @��@    @��@    @A�@    @��@    ���@    �9�@    �{�@     C�@    @>�@    @/�@    @F�@     ��@    `��@    �s�@    �
�@     ��@    �$�@    `}�@    ���@    ���@    ��@    `��@    ���@    �;�@    ��@    ���@    ��@    `o�@     �@    ���@    �u�@    ���@     ��@    ���@    @q�@    `��@     �@    @��@    @��@    @��@    ���@     ��@     U�@     ��@    �G�@     h�@    ���@      �@     ��@     �@    �z�@     W�@     ��@    �P�@     ��@     ��@     *�@     �@     �@     P�@     d�@     %�@     ��@     j�@     ��@     ��@     �@     ��@     ��@     ��@     ��@     ��@     ��@     8�@     V�@     ��@     Z�@     r�@     �@     �@     ��@     ��@     ��@     H�@     B�@     ڠ@     ��@     ��@     L�@     ��@     �@     ��@     ��@     ��@     ��@     P�@     @�@     p�@     ��@     Ę@     ��@     �@     ��@     ��@     h�@     $�@     ԕ@     ��@     ��@     ,�@     �@     �@     ��@     ��@     ��@     l�@     �@     L�@     Ȏ@     ��@      �@     @�@     �@     (�@     ��@     X�@     �@      �@     x�@     ��@     �@     ��@     8�@     @�@     ��@     ��@     ��@     �@     0�@     �@     X�@     x�@     ��@     @�@     ��@     @�@     ��@     ��@     8�@     ��@     ��@      �@     �@     ��@     �@     �@     @�@     �}@     p�@     ��@     @     �~@     �~@     ��@     �|@     `{@     p|@     �{@     �{@     �z@     �y@     px@      y@     �w@      x@     �{@     �v@     �w@     �x@      w@     �v@     �x@     �s@     �t@     �t@     @v@     �s@     �s@     �t@     �s@     s@     pr@     �q@     �p@     �p@     �p@      q@     @q@     �n@     @m@     @n@      m@     `m@      o@     @k@      j@     �m@     �m@      i@      k@     @o@     �h@     �g@     �g@     �i@     �j@     `g@     @f@      g@     �e@     �g@     @f@     �g@     �d@     �e@     �e@     �d@      h@     �a@     �`@     @`@     �_@      _@     �]@     `a@      `@     @`@     �]@     �[@     @_@     �[@     @_@     @_@     �Z@     �Z@     �^@     �Z@     �Z@     �U@      Y@      X@      Y@     �U@     @V@     @T@     �S@     �U@     @T@     @T@     �V@      S@     �S@     �R@     �Q@     �P@     �P@     �Q@     �O@     �Q@      U@     �S@     �U@     �P@      I@      O@      I@     �G@     �L@     ��@     ̪@     @R@     �T@     �R@      Q@     �R@     �P@     �S@     �T@     �Q@      S@     @Q@     �T@     �U@     �U@     �Y@     �V@     �R@     @X@      Z@     �V@      Y@     �Z@      [@     �^@     �]@     �Z@     �a@      Y@     @`@     @`@     @Z@     �`@     `a@     `a@     �]@     �[@     �a@     �a@      d@     @]@     �c@     �b@     �e@      e@     @g@     @d@      g@     �e@     �d@     �e@     �e@     `e@     �j@     `h@      g@     �h@      j@      n@     �k@     �j@     @o@      l@     �k@      i@      k@     �l@     �l@     @o@      l@     @n@      m@     �o@     �o@     @n@     0p@     Pq@      o@     `p@     Pr@      r@     �r@     �t@      v@     �w@     �w@      v@     @u@     �v@     pv@     �t@     u@     �w@     @v@     @x@     `y@     �z@     �w@      x@      y@     Pw@     @x@     Px@     �w@     `z@     �y@     pz@     �{@     0|@     pz@      }@     �@     @@     �@     �@     X�@     �|@     (�@     x�@     �@     (�@     X�@     (�@     ��@     �@     ��@     ��@     Ђ@     ��@     ��@     ��@     h�@     Ѓ@     ��@     ��@     8�@     P�@     �@     Ȇ@     ��@     `�@     ȉ@     ��@     X�@     ��@     ��@     ��@     ��@      �@     ��@     �@     0�@     ��@      �@     `�@     8�@     8�@     \�@     P�@     @�@     ��@     d�@     ԑ@     ��@     `�@     �@     \�@     �@      �@     �@     ��@     ��@     L�@     ��@     ��@     \�@     8�@     l�@     ��@     �@     ��@     ��@     ��@     ��@     @�@     �@     �@     �@     ��@     ��@     0�@     ��@     b�@     ��@     ��@     �@     ��@     ,�@     ,�@     X�@     ��@     ̧@     ��@     �@     ��@      �@     N�@     F�@     `�@     ԯ@     ��@     ��@     ��@     Ѳ@     o�@     �@     �@     ӷ@     �@     .�@     ��@     E�@     �@     ��@    ���@     �@    �5�@    ���@    ��@    �7�@    ���@    ���@    ��@    ���@    ��@    ��@    ���@     ��@     �@     5�@    �+�@    ���@     A�@    �*�@     /�@    `�@    �s�@    �n�@    �\�@    ��@    ���@    `y�@    ��@     h�@    ���@    �L�@     ��@    `V�@    �o�@     ��@    �L�@    �v�@    `�@     ��@     ��@     7�@    ���@    @��@    ���@    � �@    @�@    ��@    `��@     J�@    @��@    @.�@    �P�@     �@    @��@    ���@     t�@    @=�@     ��@    �s�@     ^�@    �Q�@    ��@    ��@    @��@    �>�@     &�@     Q�@     ��@     ��@     ��@     x�@     ��@     ��@      v@     �h@      e@     @m@     @h@        
�
predictions*�	    �ᵿ   �|�?     ί@!  Pv>D'@)�mMf6u$@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.���vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�[#=�؏�>K���7�>�uE����>�f����>O�ʗ��>>�?�s��>��[�?1��a˲?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?+A�F�&?I�I�)�(?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�������:�              �?              @      @      5@     �B@     �N@     @X@     �Z@     �f@     �c@     �g@      e@      f@      b@     @^@     �]@     @X@     @V@     �U@      I@      M@     �L@      H@      F@     �C@     �B@      ;@      >@      5@      ;@      6@      1@      3@      1@      ;@      0@      ,@      @       @      "@      "@      "@      "@      @      @       @      @      @      �?      @      @              @      �?      �?       @              @      �?      �?      �?               @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              @      @              �?              @       @              @      @      @      @      @      �?      @      �?      @       @      @      @      @      @      @      @       @      (@      $@      ,@      2@      7@      .@      (@      0@      ,@      4@      7@      6@      ;@      4@      ?@      B@      B@      A@     �G@     �C@     �@@     �C@      I@     �A@     �N@     �F@      D@     �B@     �C@     �D@      F@      B@     �G@     �C@      D@      B@     �I@     �@@      5@     �D@      =@      5@      7@      6@      2@      ,@      @      &@      @      @      @       @               @      @        1����1      }�j�	��2Y���A*�b

mean squared error�(6=

	r-squared�ӑ=
�L
states*�L	   ����   ���@   �$[RA!�O7?ab��)%qqEA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             <�@     P�@     ̐@     @�@     ��@     ��@     Ԙ@     <�@     f�@     ��@     �@     m�@     ��@     ��@    ���@    ���@     !�@     �@     ��@     ,�@    �a�@    ���@     ��@    �/�@    �V�@     ��@     0�@    ���@    �+�@    �]�@     }�@    �`�@    ���@    @q�@    �3�@    `��@    �y�@     �@    ���@    `L�@     ��@    ���@     �@    `��@     �@     @�@     ��@    `��@    ���@    ���@    `a�@      �@     ��@    @��@    `J�@     Y�@    ���@    ��@    ���@    ���@    @]�@     ��@     ��@     ��@    ���@    @C�@    ���@     c�@     ��@     ��@     v�@     ��@     ��@    ���@    ��@     u�@     �@     ��@     j�@     O�@     ��@     ޼@     غ@     չ@     r�@     ��@     W�@     ��@     ��@     ��@     ϰ@     X�@     ��@     ��@     �@     ��@     z�@     �@     ا@     ��@     �@     &�@     ^�@     t�@     ��@     ��@     
�@     ��@     ��@     ʠ@     Р@     ��@     �@     �@     ��@     (�@      �@     �@     ��@     ܛ@     X�@     �@     ��@     ��@     P�@     Ș@     `�@     ��@     X�@     �@     �@     ̕@     �@     ��@     ��@     ,�@     ��@     ��@     L�@     �@     �@     h�@     H�@     �@     p�@     8�@     x�@     �@     ��@     ��@     ��@     ؍@     X�@     H�@      �@     x�@     (�@     ȋ@     @�@     ��@     ��@     `�@     ��@     �@     ��@     H�@     ��@     H�@     ��@     X�@     ��@     ��@     @�@     ؄@     ��@     �@     ��@     ��@     H�@     ��@     ��@     Ё@     P�@     `@     ��@     ��@     @@     ��@     H�@     @@     �~@     �|@     `~@     �}@     �}@     �z@      x@     �}@     �}@     `z@     �y@     py@      y@      x@     px@     0w@     Px@     @w@     �v@     �w@     u@     �u@     �t@      r@     �r@     `s@     �t@      r@     Pq@     �r@     �r@     `o@     `q@     �q@      p@     �p@      n@     `p@     p@      n@      n@      m@     `l@      j@     �j@     @k@     `l@      i@     �g@      j@     �h@     `h@     �i@     �f@      i@     `h@     �d@     �f@     �g@     �g@     @k@     �c@     �b@     �a@     �`@     @d@     �`@     @a@     �d@     �`@      c@     �^@     �`@      a@      `@      c@     @^@     �[@     ``@      `@     �X@     �Z@     �Z@     @Y@      ]@      \@     @Z@     �^@     @U@     �Y@     �V@     �U@     �W@     �V@     �Z@     �X@     �U@     �Q@      V@     @V@     �P@     �S@     �Q@      P@     @S@      S@     �O@     �M@     �O@     �S@      P@      N@     ��@     p�@     �T@     �V@     @V@      U@     �R@     �Q@     @T@     �S@      V@     @X@     @U@      V@      S@     �R@      Y@     �]@     �\@     �Z@      X@      [@     @`@      ]@     �`@     �\@     �^@      `@     @_@      ^@      `@      b@     �c@      _@     �f@     @a@     �a@      c@      d@     �d@     �Z@      d@     �c@     �e@      f@     �e@      f@      g@     �g@     �g@     `f@     @e@     �i@     �g@     �h@      h@      l@     �h@     �l@     �f@     �h@     �k@     p@     �m@     �p@     `l@     @o@     @m@     �q@     @n@     �n@      m@     `p@     �r@     �m@     �p@     �p@     Pr@     �q@      r@     `s@     @s@     �x@     �u@     �t@     `u@     �v@     �u@     y@     �v@     �v@     Pw@     �z@     �y@     @w@     0v@     {@      �@     0x@     x@     �x@     �x@     �{@     @y@      }@     �}@     �|@     `|@     �|@     �|@     �~@     p|@     0~@     x�@     �@     X�@     ��@     P�@     p�@     H�@     �@     ��@     ��@     ��@     p�@     ��@     h�@     ��@     ��@     8�@     (�@     ؃@     P�@     ��@     �@     ��@     �@     �@     ��@     ؈@     ��@     (�@     (�@     �@     h�@      �@     ��@      �@     h�@     ��@     ��@     ��@     ��@     �@     ��@     \�@     \�@     �@     ,�@     �@     T�@     x�@     �@     d�@     ��@     0�@     �@      �@     D�@     ��@     (�@     ��@     ̕@     ��@     (�@     �@     ��@     �@     �@     �@     ԙ@     ��@     �@     �@      �@     x�@     ؞@     ̟@     �@     B�@     ܡ@     h�@     d�@     �@     \�@     ^�@     B�@     ��@     ��@     �@     ��@     �@     h�@     z�@     ��@     r�@     Ϋ@     �@     ��@     ̮@     Я@     ��@     ��@     ��@     ��@     z�@     ��@     T�@     ,�@     e�@     ƺ@     ��@     ��@     ��@     ��@     ��@    ���@     ��@     Q�@     ��@    ���@     ��@    �T�@    ���@    @F�@     ��@    � �@    @`�@    @(�@    ���@    ���@    @��@     ��@    �;�@     S�@    `j�@    �o�@    �}�@    ��@     ��@     ��@    �;�@    @��@    �C�@     ��@    `=�@    @��@     �@    ���@     ��@     ��@     ��@     ��@    �D�@     ��@    ���@     -�@    �E�@    ���@    ���@    `t�@    ���@    ���@    @�@    ���@    �r�@     ��@    ���@     �@    ���@    @a�@    �g�@     ��@     �@    ���@     ��@     ��@    ��@    @��@     �@     ��@    ��@     �@     ̡@     ��@     ��@     ��@     �@     Ȁ@     0|@      o@     �e@     �q@     @h@        
�
predictions*�	   ��Ѹ�    e�?     ί@!  چ��3�)�;����'@2�
%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���bȬ�0���VlQ.��7Kaa+�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1����%�>�uE����>>�?�s��>�FF�G ?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�������:�
              �?      @      "@      $@      ?@     @V@     �V@     @e@     @f@     �d@      f@     `e@     �f@      c@     �b@      ^@     �^@     @W@     �T@      R@     @R@      I@     �R@      F@      N@      D@      E@      B@      B@      A@      :@      6@      8@      4@      1@      .@      *@      *@      @      @      $@      *@      @      @      @       @       @      @      @      @      @       @       @               @      �?      @       @       @      �?               @      �?              �?      �?              �?      @      �?              �?              �?              �?              �?       @               @      �?      �?               @      @              �?      �?      �?              �?      �?       @              @      �?      @      @       @       @       @              @      @      @      @      @      @       @       @      (@       @      3@      0@      0@      .@      0@      7@      0@      (@      ,@      3@      =@      2@      A@      6@      C@      :@      A@     �A@      A@     �D@      B@      @@      :@      B@      D@     �B@      >@     �C@     �A@     �B@      :@      8@      ;@      =@      7@     �C@      5@      .@      5@      2@      0@      (@      @      "@      $@      @       @      @      @       @      @      @      @       @       @              �?        S囷b1      k�		���Y���A*�b

mean squared error��6=

	r-squared�k�=
�L
states*�L	   @���   @u�@   �$[RA!�s�E9��)��X}��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             4�@     ��@     ��@     t�@     ̗@      �@     ��@     Ȟ@     "�@     ��@     M�@     ݻ@     ��@    �:�@    �F�@    @��@     <�@    ���@    ���@    ���@     ��@     M�@    @��@     �@    �A�@     ��@    ��@     ��@    �i�@    ���@    @m�@    @��@    `b�@    �&�@    @��@    ���@    `U�@     �@     ��@    `.�@    `��@     ��@    @
�@    ���@     �@    ���@    `��@    �b�@      �@    ��@     l�@    �+�@    �_�@    `�@     W�@    ���@     ��@    `��@    `��@    ��@    @��@     �@    @��@    @=�@    @P�@    ���@    ���@    @��@     \�@     ��@     X�@     ��@    �0�@     R�@    �d�@     ��@     ��@    ���@     �@    �^�@     �@     �@     �@     V�@     ��@     �@     ��@     ��@     �@     �@     �@     6�@     ��@     ��@     ��@     :�@     d�@     R�@     @�@     �@     ��@     ��@     ��@     ��@     "�@     ҡ@     ʡ@     �@     �@     �@     ��@     �@     �@     ��@     L�@     @�@     ��@     0�@     ܛ@     ��@      �@     ؛@     ��@     Ș@     �@     ��@     ��@     ,�@     ��@      �@     ܕ@     ��@     L�@     H�@     (�@     l�@     �@     �@     8�@     ��@     �@     �@     <�@     <�@     l�@     `�@     ��@     H�@     ؏@     ��@     �@     (�@     0�@     `�@      �@     8�@     ��@     x�@      �@     ��@     P�@     ��@     ��@     ��@     �@     ��@     `�@     ��@      �@     @�@     H�@     @�@     ��@     X�@     h�@     ��@     ��@     (�@     ��@     �@     8�@     �@     Ȁ@     ��@     �~@     @�@     ؀@     (�@     ��@     �@     ��@     @@      �@     �}@     P}@     0}@     �{@     �z@     @{@     `x@     py@      x@     �z@      x@     �v@     �v@     �w@     pv@     0x@     @v@     �s@     �u@     �v@     �t@     �r@     �s@     �s@     ps@      t@     �r@     �q@      r@     �r@     �q@     �l@     �m@      m@     �p@     `p@     t@     q@     @o@     �p@     �l@     �p@     �l@     �m@     �h@     �k@     �h@     �j@     �g@     `i@     @i@     @k@     �e@     �e@      h@     @d@     �c@      g@      e@     �g@     �d@      g@     `d@      f@     �d@      f@     �c@      c@     �e@     @c@     @c@     �Z@     �a@     �a@     �`@      _@      ^@     @_@     �\@     �[@      ]@     �[@     �[@      Z@     @^@     �]@     �V@     @[@     @Z@      Y@     �W@     �Y@     �U@     �V@     �T@      W@     �X@      Z@     �T@     �S@      Y@     @T@     �P@     �O@     �T@     �S@     �V@     @R@     @P@     0�@     %�@     @X@     �S@     �W@     @V@     �T@     �V@     �Y@     @W@      V@      ]@     �Y@      a@      ]@     @Y@     �X@      X@     �\@     �Y@     @[@      _@     �^@     @Y@      ]@      a@     �^@      _@     �`@      b@      `@     �a@     `b@     �_@     �a@     �`@      `@     @a@      c@     �b@     @f@     `f@     @c@     �b@     �e@     �f@     �g@      i@      f@     �f@     `e@     �f@     �g@      h@     �j@      j@     �j@     �h@      l@     �i@     �k@     �n@     �k@     �m@     �l@     �p@      o@     pp@     `n@     q@     �t@      p@     �p@     �q@     s@     @q@      r@      t@     0q@     ps@     �s@     `v@     `u@     `s@     �t@     �r@     �u@     0u@      w@     �u@      y@     `w@     0y@     px@     �x@     x@     �z@      |@      z@      y@     �y@     �x@     pz@     {@     @|@     @}@      |@     P|@     �z@     �}@     �{@      }@     `~@     ��@     x�@     0�@     ��@     �@     �@     0�@     ��@     ؂@     ��@     x�@     ؂@     `�@     h�@     ��@     ��@     ��@     ؄@     ȃ@     ��@     ��@     H�@     �@     0�@     (�@     p�@     X�@     ��@     Ј@     �@     h�@     p�@     0�@     ��@     �@     Ќ@     ��@     �@     H�@     ؎@     0�@     p�@     Ԑ@     ��@     L�@     �@     ��@     `�@     x�@     ��@     Ē@     В@     �@     ȕ@     �@     4�@     �@     Ė@     ��@     �@     h�@     x�@     ��@     4�@     d�@     ��@      �@     �@     ��@     ��@     ��@      �@     �@     ��@     ��@     ��@     �@     ��@     �@     `�@     ��@     ��@     ��@     ̣@     4�@     d�@     <�@     ��@     �@     V�@     �@     Ȩ@     >�@     *�@     ̬@     ��@     ^�@     ��@     İ@     =�@     h�@     �@     ��@     )�@     j�@     ۷@     ߸@     4�@     ޻@     P�@     ��@    ���@    ���@    �.�@    �x�@     ��@     �@    ��@    �J�@     �@     ��@    �"�@    @u�@    �b�@    @��@     f�@    @d�@     "�@    ���@    ���@     u�@     ��@    ���@    �a�@    �p�@    @#�@    ���@     _�@    ��@    �X�@     ��@    �_�@     ��@    ��@    ���@     ��@     F�@    �:�@     8�@    `�@    ��@    ��@     ��@     �@    �j�@    `��@    �g�@    ���@    ���@    ���@    `��@    ���@    ���@    @]�@     ��@    ���@     L�@    �Y�@     T�@    ���@    ���@     ��@     �@    �Y�@    �i�@    @��@    @�@     q�@    @��@     g�@     x�@     ��@     ,�@      �@      �@     0�@     p~@     @u@     �s@     �q@     `g@        
�
predictions*�	   �����    �c�?     ί@!  dd�X8@)��{W�%4@2�
8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !�ji6�9���.����ڋ��T7����5�i}1�6�]���1��a˲���ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�
              �?              @      $@      >@     �P@     �]@     @d@     `i@      i@     �e@     @d@     �`@     `b@     �^@      X@     �U@      T@     @R@      P@     �O@     �H@      F@      @@      >@     �@@      ?@      =@      5@      >@      4@      3@      0@      1@      .@      "@      @      @      @       @      @      @      @      @      $@      �?      �?      @      @      �?       @       @      @      �?       @      �?       @              �?              �?              �?              @              �?      �?      �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              @               @       @      @      @      @              @      @       @       @      �?      @      @      @      @       @      @      "@      @      (@      "@      "@      "@      @      @      3@      2@      .@      2@      5@      2@      <@      <@      8@      =@      H@     �@@      :@      B@     �B@     �F@      H@      C@      B@      J@      K@     �D@      J@     �I@     �G@      E@     �E@      I@     �A@      I@      :@     �C@      B@      ?@      <@      4@      5@      6@      4@      6@      (@      *@      @      &@      $@      "@      "@      @      @      @      @      @      �?      @               @        �� R�2      ���	��Z���A*�e

mean squared error��6=

	r-squared�`�=
�L
states*�L	   �\��    ��@   �$[RA!�:��h��)�1�b�aA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             (�@     <�@      �@     \�@     �@     ��@     ��@     X�@     V�@     ~�@     �@     A�@    ��@     6�@    �E�@     ��@    ���@     ��@    �(�@    ���@    �z�@    ��@    �[�@    @��@    @��@    ���@    ���@    ���@    ���@    ���@     ��@     g�@    @��@    `��@    �C�@     ��@    ���@    �n�@    �I�@    @�@    ��@    @�@     ]�@     ��@    @��@    ���@    ���@    `�@    `��@     ��@    �g�@    �9�@    ���@     �@     ��@    ��@     m�@     ~�@    ���@    @��@    ���@    @��@    @{�@    �Q�@    ��@     ��@    ��@     ��@    @�@     ��@     x�@     |�@    ��@    ��@    �o�@    �\�@     ��@     x�@     2�@    ���@     �@     ڽ@     ��@     �@     �@     x�@     ��@     ��@     ߳@     ʲ@     ��@     \�@     &�@     ��@     ¬@     ��@     ��@     �@     ��@     `�@     ,�@     Ȥ@     ��@     ��@     ��@     �@     x�@     ��@     d�@      �@     X�@     �@     P�@     ��@     ��@     H�@     ��@     �@     ,�@     ��@     H�@     �@     �@     ��@     L�@     ��@     p�@     $�@     P�@     $�@     ��@     ��@     D�@     D�@     h�@     �@     Ȓ@     ��@     P�@     Ԓ@     ��@     d�@     ȏ@     $�@     X�@     ��@     ��@     ��@     ȋ@     p�@     �@     �@     p�@     ،@     Ћ@     Ћ@     ��@     ��@     @�@     P�@     (�@     `�@     ��@     ��@     ��@     ؈@     ��@     8�@     h�@     ��@      �@     �@     �@     ؅@     ��@     ��@     ��@     ��@     ؁@     h�@     0�@     ��@     ��@     H�@     �~@     p~@     `}@     �@     �~@     �@     }@     �}@      {@     �{@     `|@     P|@     �z@     �}@     `{@     x@      y@     �y@     �v@     pv@     `v@     �t@     �w@     �w@     �w@     @w@     pu@      t@     @s@     �t@     �q@     �t@     �q@     `q@     r@     �r@      r@     pq@     �q@     �q@     `p@     p@     @p@     �q@     0p@     �m@     �m@      m@      n@     @j@     @m@     �k@     `m@     @k@     @k@      i@     @i@     @i@     @i@     �n@     �f@      f@     @f@     �f@     �e@     @d@     �f@     �g@     �c@     `b@     @g@     `a@     �`@     @c@     �c@     �b@     �a@     @c@     �d@     @b@      a@      _@     @]@     �_@     �d@     `c@     @^@     @]@     �]@     �[@     �`@     �[@     �]@     �]@     �T@      ]@     @Z@     �X@     �Y@     @Z@      U@     @Y@     �V@     @V@      Z@     @Y@     @V@      T@     @W@      S@     �P@     @S@      U@     @T@      T@     �Q@     �R@     @S@     ��@     ��@     �T@      S@     �Y@     @U@     @S@     @U@     �U@     @R@      Z@     �W@     �X@      Y@     �Z@      [@     �X@     �Z@     @[@     �_@     �W@     �`@     �X@     @X@     @Z@      ^@     �\@     �a@     �^@     �Z@     `a@     �^@     �a@     �`@     �`@     `a@      a@     �b@     �c@     `a@     @d@     �c@     @h@     �i@      l@      e@     �d@      h@      g@     �e@     �h@     �i@      f@     �g@      g@     `m@      i@     @m@     @k@      j@     `h@     `j@     �l@      o@     �k@     �l@      o@     @m@     `m@     �n@     `m@     `n@     �o@      m@      p@     @o@     @p@     p@     @q@     �q@      v@     �s@      u@     �t@      t@     pu@     Pt@     @u@     0u@     �u@      v@     0y@     �v@     �u@     �v@      y@      u@     `w@     �x@     0x@     �w@     {@     �|@     0|@     �|@     �y@     �|@      z@     �}@     �{@     `z@      |@     @z@     �|@     �}@     �~@     P~@     @}@     `~@     H�@     ~@     X�@     ��@     ��@     X�@     ��@     ��@     H�@     �@     ؂@     ��@     `�@     ��@     ��@     ��@     �@     Ѕ@      �@     ��@     ؆@     ��@     ��@      �@     ��@     Ȋ@     `�@     p�@     ��@     8�@     ��@     h�@     ��@     `�@     8�@     H�@     ��@     ��@     �@     �@     ԑ@     ��@     ȑ@     l�@     l�@     ��@     (�@     ��@     ԓ@     �@      �@     (�@     ��@     �@     l�@     @�@     \�@     l�@     ��@     ��@     ��@     ��@     �@     (�@     l�@     ��@     �@     (�@     x�@     t�@     ��@     $�@     4�@     x�@     �@     .�@     �@     �@     4�@     4�@     ʣ@     ��@     �@     �@     ��@     ��@     ��@     Ԫ@     ��@     :�@     $�@     0�@     t�@     �@     ��@     .�@     1�@     {�@     ��@     �@     ظ@     ��@     0�@     ��@    ��@     �@     t�@      �@    ���@     ��@     �@    ���@     {�@    ��@    @��@    ���@    ���@    �S�@     ��@    @��@    ���@    ���@    �.�@     ��@     ��@    @�@    ���@    `��@     w�@    `
�@     f�@    ���@    ��@    ���@    ���@    ���@    ���@    @�@    @O�@     |�@    �j�@    @��@    `7�@     ��@     )�@    @=�@    `��@    ��@    �Q�@    ��@    @:�@    `��@    �&�@    ���@     ��@    ���@    @��@    @~�@    @"�@     ��@    ��@    �1�@    ���@    ���@    @d�@    ��@    ���@     �@    ���@    ���@     ��@     ��@    ���@     ��@     ��@     ��@     ��@     ,�@     Ȍ@     �@     `�@      v@     �u@     �@     @l@        
�
predictions*�	   ��E��   @���?     ί@!  Pz��@){����@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�1��a˲���[��>�?�s���O�ʗ���I��P=��pz�w�7���XQ�þ��~��¾����>豪}0ڰ>�ߊ4F��>})�l a�>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�������:�              �?      �?       @      @      @      &@      9@     �G@     �O@     �W@     @Y@     @\@     @a@     �a@     `b@     �b@      a@     @_@      ^@     �V@     �Y@     �R@      Q@     @T@      N@      H@     �D@      =@      9@      B@      :@      D@      4@      0@      2@      1@      4@      *@      6@      .@      (@      &@      @      &@      @      @      @      "@      @      @       @      @      @       @      @      @      �?       @       @              �?      �?       @      �?               @              �?       @              �?              �?              �?              �?              �?              �?              �?              �?      �?              �?              �?              �?      �?               @       @              �?      �?              �?              @       @       @      @      @      �?      @      @      @      �?      @       @       @      @      @       @      @      @      "@      $@      @      @      *@      .@      *@      3@      7@      2@      <@      =@      ;@      =@      5@      @@      ?@     �F@      E@      C@      F@     �I@      K@      H@     �H@     @P@     �I@      R@      N@      I@      M@     �M@     �G@      J@      >@     �D@     �B@      @@     �D@      :@      4@      *@      0@      6@      &@      $@      @       @      "@      @      $@       @              �?      �?       @       @               @      @       @      �?              �?              �?      �?              �?        �ӕ��2      ���C	��Z���A*�e

mean squared error]3=

	r-squared��=
�L
states*�L	    ���   �p�@   �$[RA!��vO`��)�e�=sZA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             0�@     ��@     H�@     �@     ��@     Ҡ@     \�@     ��@     ��@     ԩ@     گ@     ��@     0�@    �^�@    �b�@     ��@     ��@     L�@     �@    �$�@    �L�@    �Z�@    @��@    @b�@     ]�@    @��@    ���@    ���@    �#�@    `��@    �m�@    ���@    `��@    `��@     N�@    ���@    �S�@    �6�@     ��@     ��@    @��@    �c�@    �x�@    ���@    `��@    ���@    @��@    @��@    �2�@     ��@    �N�@    ���@    `��@     K�@    ���@     7�@    �?�@    @n�@    ���@    ���@    @	�@     ��@    �i�@     ��@    @B�@    ���@     K�@     ��@    �L�@    �c�@    �	�@    ��@    ���@    ��@    ��@     ��@     Q�@     	�@    �Z�@     C�@     �@     ,�@     4�@     ��@     ��@     D�@     ��@     ,�@     9�@     *�@     �@     ̱@     k�@     ��@     ��@     �@     ��@     ��@     ��@     T�@     ��@     ��@     ��@     r�@     �@     ��@     أ@     ��@     �@     ��@     ޡ@     r�@     �@     L�@     �@     l�@     ̞@     @�@     �@     P�@     ��@     @�@     �@     ��@     p�@     ��@     �@     X�@     З@     ��@     T�@     X�@     L�@     x�@     ��@     ��@      �@     �@     `�@      �@     \�@     �@     �@     |�@     0�@     ��@     ��@     ��@     p�@     Ȑ@     �@      �@     �@     �@     @�@     8�@     �@     ��@      �@     P�@      �@     ��@     ؍@     ��@      �@     ��@     `�@      �@     �@     ��@     ��@     `�@     �@     h�@     8�@     ��@      �@     ؃@     `�@      �@     ȃ@     X�@     @�@     8�@     ȁ@     �@     8�@     h�@     �~@     `~@     �@     ��@     �@     �~@     �{@     p|@     �{@     0|@     P{@     �z@     `|@      {@     0z@      x@     �v@     �z@      {@     @w@     w@      v@     pu@     �z@     �v@     �u@     �w@     �w@     �t@     �t@     @t@      r@     �s@     `r@     �q@     �p@     s@     �q@     �n@     �q@     0r@      q@     �p@     �p@     `o@     �p@      r@     �p@      o@      m@      m@     �p@     `m@     �h@     `k@     `l@      l@     �i@     �i@     `j@     �i@     @h@      h@     @g@     `g@     �g@     @l@     `h@     �g@      d@     �d@     `d@     `c@     �b@     �b@      a@     `a@     �a@     @^@     @c@     �a@     `d@      c@     �d@     �c@      a@     @_@     �_@     @_@     �[@     �Z@     �Y@     ``@     �[@     �Y@     �[@     �X@     @Z@     @^@      U@     �V@      [@     �W@     �X@     �Y@     @X@     @S@      W@     �W@     �W@     @V@     @V@     @R@     �X@     Ӳ@     5�@     @V@      X@      Y@     �Y@     �[@      Z@      Z@     �[@     �W@     �[@     @U@      [@      ]@      ]@      ^@     ``@     �^@      X@     �a@     �Y@     �`@     ``@     �^@      b@     �d@      b@     @f@      c@     `e@     �d@     �b@     �a@     @d@     @b@      d@     �c@      f@      d@     �c@     �e@      f@     �h@     �g@     @g@     �j@      i@     @h@     �i@     @j@     �i@     `k@      l@     �h@     @l@      k@     �k@     �l@     �m@      p@     �n@     �l@     �l@     �m@     0p@     �o@     @m@     �p@     Pp@      p@     0p@      q@     `q@     ps@     pr@     �q@     �s@     �s@      x@     `u@     �q@     @s@     �v@     �v@     `w@     w@     pu@     Pu@     0v@     �y@     0u@     �y@     Px@     py@      y@     �y@     �y@     �y@     �y@     0|@     0{@     �{@     P~@     �}@     �{@     �}@     0~@     0}@     (�@     �~@     �@     8�@     ��@     P�@     ��@     X�@     (�@     P�@     �@     ��@      �@     ��@     0�@     ��@     H�@     Ђ@     0�@     �@     x�@     h�@     P�@     �@     0�@     ��@     h�@     ��@     ��@     h�@     x�@     (�@     ��@     ��@     p�@     ��@     `�@     X�@     ��@     0�@     ��@     �@     P�@     <�@     ��@     ,�@     ��@     @�@     ��@     D�@     �@     t�@     ��@     H�@     ��@     В@     p�@     ܔ@     p�@     ��@     $�@     D�@     \�@     t�@     ��@     ��@     (�@     ��@     ��@     X�@     L�@     ��@      �@     ܚ@      �@     ԝ@     Ȟ@     P�@     ğ@     ��@     ��@     2�@     ��@     �@     6�@     v�@     £@     ��@     �@     6�@     L�@     �@     Z�@     ��@     ��@     ��@     <�@     ��@     *�@     �@     F�@     ��@     ��@     Բ@     ͳ@     0�@     �@     ��@     ��@     ��@     �@     L�@     ��@    ��@     `�@    ���@     ��@    ��@    ���@     �@     *�@     e�@     ��@    @b�@    @��@    ���@    @V�@     )�@    ���@    @��@    ��@     ��@    @y�@    ��@    ���@    ���@    `��@    �M�@     4�@    ���@    @c�@    ��@    @t�@    ��@    `0�@    ���@    ���@    �<�@    ���@    `u�@    ���@    ���@    @P�@    `��@     <�@    ���@    ��@     ��@    �R�@     ��@    `��@    �(�@    @��@     n�@    @m�@    @y�@     ��@    @�@    @�@     ��@     ��@    �5�@    �.�@     �@    ���@    @[�@    ���@     S�@    �6�@    �.�@    �&�@     \�@    @�@     R�@     F�@     �@     ��@     �@     ��@     P�@     x�@      ~@     �{@     8�@     ��@        
�
predictions*�	   �?K��   `�U�?     ί@!  U�U@) �4�"�2@2�Ӗ8��s��!������8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��5�i}1���d�r������6�]���1��a˲���[���FF�G �pz�w�7��})�l a�iD*L�پ�_�T�l׾K+�E���>jqs&\��>1��a˲?6�]��?�5�i}1?�T7��?�vV�R9?��ڋ?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?�������:�              �?              �?       @      @      @       @      "@      (@      2@      6@      7@      @@      C@     �F@      M@      G@     �O@     �K@      P@     �I@     �Q@      H@     �E@      L@      K@      G@      ;@      ?@     �D@     �D@      8@      7@      2@      =@      9@      2@      1@      .@      @      "@      $@      "@      @      $@       @      @      @      @      @      $@      @      @      @      @       @      �?      @       @      @       @      @      �?              �?              �?      �?      �?      �?       @      �?              �?               @              �?              �?      �?              �?              �?              �?              �?              �?               @              �?      @      �?      �?              �?               @      �?      �?      �?              @      @      @      @      @      @      @      @      @      @      &@      *@      @      &@      @       @      &@      *@      0@      3@      1@      <@      <@      4@      @@      ?@      A@      B@     �D@      F@     �G@      N@     �N@     �D@     @S@     �Q@      P@     �S@     �Z@     �X@     �U@      V@     �V@     �X@     �X@      \@     �X@     �Z@     �V@     �T@      S@     �S@     @Q@      P@      G@     �G@      ;@      >@      1@      1@      8@      5@      "@      ,@      @      @      &@      @      @      @      �?       @      @      �?       @       @               @              �?      �?              �?              �?      �?        는S21      "���	��[���A*�b

mean squared error�\?=

	r-squared`��<
�L
states*�L	   �H��   @��@   �$[RA!�_mI2��)�=�T�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     L�@     8�@     �@     0�@     �@     4�@     >�@     X�@     ��@     �@     ��@    `��@     ��@    �%�@    �S�@    ���@    ���@    @��@     ��@    @��@    ���@     "�@    ���@    ���@    ���@    @�@    �+�@     ~�@    ���@    �R�@     ��@    ���@    �A�@    ���@    ���@     U�@    ���@    ���@     -�@    ���@    �q�@    �9�@    ���@     �@    ��@    `��@     ��@     [�@     _�@    @Z�@    @��@    `��@    @�@    �h�@    ���@    �Z�@    ���@    ` �@    �d�@    ��@     $�@    ��@    ���@    @��@    ���@    �'�@    ��@    @��@    @��@    �i�@    ���@    ���@    ���@     !�@    ���@     R�@    �[�@    �`�@    ��@     4�@     �@     �@     غ@     �@     �@     O�@     �@     �@     �@     l�@     !�@     ��@     �@     ��@     ��@     @�@     ��@     �@     ��@     �@     `�@     ��@     ��@     ȣ@     ��@     �@     ��@     R�@     ��@     R�@     ��@     ��@     �@     x�@     �@     p�@     (�@     ��@     ��@     ę@     ��@     ��@     ��@     Ԗ@     �@     ̖@     `�@     L�@     ��@     ܕ@     <�@     <�@     T�@     T�@     t�@     ��@     d�@     P�@     �@     ��@     d�@     ��@     ��@     d�@     �@     �@     H�@     ��@     X�@      �@     ��@     ��@     ��@     @�@     @�@     (�@     ��@     h�@     ��@     �@     ��@     ��@     ��@     ��@     ȇ@     ��@     �@     ��@     ��@     `�@     ȇ@     ��@     ��@     ȅ@     Ȅ@     ��@     ��@     P�@     P�@     0�@     h�@     ��@     ؁@     p�@     p�@     �@     �@      �@      �@     `�@     8�@     �@     �~@     p�@     �}@     ~@      |@     0|@     0{@     ~@     p{@      z@     �x@      z@      x@     w@     �y@      x@     pv@     �z@     u@     �u@     �t@     �s@      u@     @u@     pt@     �t@     `s@     �t@     `r@     �q@     s@     �q@     �s@     `p@     �p@     �q@     �p@     �p@      k@     �o@     pq@     �q@     �m@     �m@     �i@     �o@     �i@      k@     `n@     �p@     0q@      l@     �h@      i@      k@     �f@     �g@     @i@      g@      f@     �e@     �d@     �f@     `e@     �f@      e@     �g@      d@     �a@     �c@     �c@     `e@     �b@     @d@     �d@      d@      `@     @`@     @b@     �`@     �a@     �c@     �`@      ^@     �Y@      ]@     �`@     �^@      \@     �^@     �`@     �Z@     �\@      [@     �]@     �W@     �_@     @X@     @Y@     �X@      X@     �[@     @W@     �S@     �U@      V@     �V@     �R@      Q@     ^�@     ��@     @R@     �Z@     @T@     �U@     @T@     �Y@     �Y@      U@      W@      X@      ^@     �Z@      ]@     @X@     �[@     �[@     �[@      ^@     �\@     @]@     �a@     �a@     �b@     @`@     @a@     �^@     �`@     `b@     �]@     �a@     `b@     `e@      d@     �b@     �c@     �a@      c@     �d@     @a@     �d@      d@     �c@     `d@      f@     �h@     @h@     `e@     �g@     �i@     �g@     �f@     �f@     �i@     �g@     @k@      i@      o@     �m@     �i@     �k@      n@     �l@     `k@     @n@     p@     �o@     @m@      m@     �p@     @n@     `q@     �o@     �q@     �p@     Pr@     �u@     �r@     �v@     �t@     �s@     �t@     �t@     0s@     Ps@     �t@      u@     �t@      v@     �u@     �y@     �v@     �v@     �x@     �w@     �x@     �v@      z@     Pz@     �y@     0z@     �}@      {@     @{@     �|@     @{@     �{@     �|@      @     h�@     �~@     �}@     �}@     Ѐ@     P�@     ��@     h�@     0�@     ��@      �@     (�@     0�@     Ȃ@     �@     ȃ@     ��@     P�@     p�@     H�@     ��@     ��@     ��@     �@     ��@     h�@     H�@     P�@     8�@     ��@     ��@     ��@     X�@      �@     ��@     Ќ@     �@     X�@     ��@     8�@     H�@     ��@     �@     l�@     �@     �@     �@     �@     ��@     l�@     �@     �@     Ē@     �@     �@      �@     ��@     h�@     Ȕ@     ��@     (�@     (�@     ��@     ��@     �@     d�@     �@     ��@     ��@     ��@     H�@     �@     �@     ��@     0�@     L�@     ��@     T�@     4�@     0�@     �@     ȡ@     ��@     �@     X�@     ̣@     ��@     H�@     �@     ��@     ʥ@     x�@     p�@     �@     �@     �@     ��@     z�@     �@     �@     ��@     ��@     s�@     5�@     x�@     �@     �@     �@     �@     G�@     ��@     ��@     �@    �&�@    �V�@     ��@    ��@     ��@    �/�@    ���@    �q�@     ��@    ���@    ��@    �6�@    @D�@    ��@    ���@    ���@     n�@    @�@    @��@    ���@    ���@     k�@    @(�@    ��@    �X�@    ���@    �@�@    `��@     ��@     ��@    ��@     �@    �|�@    ���@    ���@     ��@    @M�@    `��@     x�@    ���@    �E�@    @$�@    @��@    ���@    ���@    �5�@    ���@    �%�@    ���@    `��@     "�@    ���@    @��@    @�@    @s�@    ���@     �@     }�@    �C�@    ���@    �Z�@    �7�@    �M�@     ��@    �D�@     "�@    ��@     ��@    `3�@     ��@     �@     ��@     h�@      �@      �@     ��@     ��@     �}@     P@     ��@     ��@        
�
predictions*�	   �����   �&T�?     ί@!  B��M�)Z�$��@2�
!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'��1��a˲���[����[�?1��a˲?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?�������:�
              �?      �?              �?      @       @      @      @      ,@      =@      E@     �R@     `a@     �h@     @j@      q@     �q@     0t@     �s@     @p@     �n@     @k@     �d@     �a@     �X@      T@     �S@      K@     �H@     �B@      A@      6@      9@      4@      2@      0@      .@      *@      @      @      @      @      @      @      @      �?      @      @      @      "@      @       @       @       @      �?      �?       @      �?      �?              �?      �?              @      �?              �?      �?              �?      �?      �?      �?              �?              �?              �?              �?              �?      �?      �?               @       @               @               @       @      �?              �?      @      @      @      @      @      @      @      @      @      @      @      @      @       @      @      @      &@      @      @       @      $@      "@      &@      &@      $@      *@      @      &@      ,@      *@      2@      (@      :@      4@      @      (@      "@       @       @      (@      &@      (@      @      @      @      @       @      @      @      @       @      @      @       @      �?      @              @       @      �?      @      �?               @      �?               @              �?        TQ�0      $aY	T�[���A*�`

mean squared error�i/=

	r-squared�F�=
�L
states*�L	    ���   ���@   �$[RA!���勺��)j��ңA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             b�@     ��@     ��@     ��@     d�@     B�@     `�@     Ԣ@     J�@     6�@     ��@    ���@     u�@     #�@    @��@    �d�@    @v�@    ���@    ���@    @�@     ��@    @P�@    �P�@    �l�@     �@     ��@    �6�@    @�@    �C�@    �X�@    ��@    �=�@    �~�@    ���@     ��@    ��@    �N�@    ��@    `�@     N�@    ��@    `��@    �N�@    �k�@     &�@    �d�@    ���@    ���@    `��@     ��@    ���@    `�@    ��@    ���@     k�@    ��@     ��@    ���@    @��@    �'�@     r�@    `��@    ��@     |�@    �e�@    @�@    ���@    �u�@    @�@    ���@     w�@    � �@    ���@    ���@    ���@     ��@    ���@     .�@     ��@    �$�@    ���@    ���@    ��@    �$�@     �@     ��@     v�@     W�@     ��@     )�@     v�@     ��@     Գ@     ٲ@     ۱@     ��@     ��@     ڮ@     Э@     .�@     $�@     ��@     �@     ��@     ֧@     �@     .�@     ��@     r�@     �@     4�@     �@     �@     �@     j�@     @�@     �@     �@     ��@     \�@     d�@     X�@     ��@     \�@     ؘ@     ��@     �@     �@     $�@     ��@     Ԗ@     ��@     ��@     ��@     D�@     $�@     ��@     ��@     P�@      �@     ԑ@     ��@     4�@     ��@     $�@     P�@     ��@     ܐ@     T�@     ��@      �@     �@     0�@     p�@      �@     ��@     x�@     X�@     (�@     P�@     ��@     x�@     ؈@     ��@     Ȍ@     Њ@     ��@     �@     8�@     8�@     X�@     (�@     8�@     �@     Ȅ@     `�@     p�@      �@     ��@     ��@     `�@     (�@     Ȃ@     Ѓ@      �@     h�@     ��@     @�@     ��@     �@      �@     �@     Ȁ@     0�@     P~@     ��@     ��@     �@      |@     �{@     p{@     p{@     �z@     0{@      |@     �{@     @|@     P{@     �{@     �x@      z@     pv@     Px@     @v@     Py@     @w@     �v@     �v@     �w@     �t@     �u@     Ps@     �t@      t@      s@     �q@     �r@     pq@      t@     @s@     0p@     �q@     @n@     Pp@     p@      q@      n@     @o@     @l@     �n@     �m@     �j@      n@     �n@     �n@     0p@      l@      i@      k@     �i@     �h@     �i@      i@     `d@     `i@      k@     �f@     �e@     @g@     @d@      c@     `c@     �l@     �k@     �d@     �e@      b@     `d@     �d@     @i@     �a@      e@     @d@     @a@     �g@     �b@     �c@     @`@     �a@      \@      _@      a@     @_@     @]@     �Y@     �Y@      Z@     �_@     �Z@     �\@     �\@     �]@     �Y@     @Y@     @Y@      Y@      \@      V@      Y@      Z@      U@     �Y@     �@     ��@     @a@      [@     @Y@      ]@     @W@     @W@     @[@      Z@     �Z@     �\@     @[@      \@     @_@      \@      [@     �_@     �^@     �[@      a@      _@     `b@      ]@     �a@     �c@     �f@     �d@     �b@      a@     �a@     @c@      a@      c@     @e@      h@     `h@     `f@     �g@      d@     `d@     �c@     �e@     �j@     �g@     @g@      i@     �g@     �e@      e@     `h@     �i@     `l@     `g@      i@     `k@     �i@      n@     �l@     �k@     �g@      j@     0p@     �p@      l@     �h@     �o@      p@      p@     �q@     p@      r@     �q@     �r@     `v@      q@     �r@     Ps@     pt@     �t@     pr@     pu@      v@     �t@     ps@      v@     �u@     �t@     �u@     �v@     �v@     �v@      y@      x@      w@     �v@      {@     �{@     �{@     @@     �{@     �z@     �z@     {@     p|@     0}@     P~@     �@     p�@     `�@     H�@     8�@     ��@     ��@     �@     ؁@     �@     p�@     0�@     ��@     ��@     ��@     h�@     ��@     �@     ��@     �@      �@     ��@     ��@     (�@     �@     �@     h�@     X�@     �@     ��@     ��@     ؊@     ��@     ȋ@     ��@     X�@     Ќ@     Њ@     �@     ��@     �@      �@     ��@     x�@      �@     ��@     0�@     (�@     ��@     x�@     ��@     �@     4�@     l�@     t�@     ��@     L�@     H�@     ��@     ��@     @�@     X�@     ��@     �@     0�@     ̗@     ��@     �@     \�@     ܚ@     �@     ��@     �@     x�@     ��@     ��@     �@     �@     �@     �@     J�@     ̠@     �@     �@     �@      �@     ��@     ��@     �@      �@     ~�@     T�@     `�@     T�@     ت@     ��@     X�@     J�@     ��@     {�@     �@     X�@     	�@     ʳ@     X�@     ֵ@     s�@     ��@     =�@     ��@     I�@     ��@    ���@    �y�@    ���@     ��@    ���@    �!�@     (�@    ���@    �m�@     ��@    �5�@    �|�@    @��@    ���@    �]�@    @��@    @O�@    @��@    @��@    ��@    @��@    ���@     ��@    `6�@    ���@    `��@    `�@    `{�@    ��@    `��@    ��@    �g�@    `�@    `h�@    ���@    ���@     ��@    ���@    �z�@     �@    `_�@    ���@    ��@    `i�@    ���@    @d�@    ���@    `��@    ���@    �s�@    @t�@    @.�@    ���@     i�@    ���@    ���@     ��@     ?�@     ��@    @��@     .�@    ���@    @��@     ^�@     ��@    @��@    ���@     ��@     ��@    @K�@     g�@    @�@    @��@     ~�@     �@     <�@     H�@     Ĝ@     p�@     ��@     Ћ@      @     (�@     P�@     �@        
�
predictions*�	   `E��   �Ϋ @     ί@!  )ڪ`@)��''VN/@2�	�v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C����#@�d�\D�X=���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(��FF�G ?��[�?f�ʜ�7
?>h�'�?�T7��?�vV�R9?�.�?ji6�9�?+A�F�&?I�I�)�(?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?\l�9�?+Se*8�?������?�iZ�?+�;$�?cI���?ܔ�.�u�?��tM@�������:�	               @       @               @       @      @      @      @      @       @      @      @      @      @       @      @      @      @       @      @      @      @       @      @      @      @      @      @       @      @              @      @      �?      @      �?      �?       @      �?       @       @              �?              �?      @      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?       @      �?      �?      @              �?              @       @      @      �?       @       @      @               @      @       @      "@      @      @      @       @      @       @      &@      @      6@      1@      1@      5@      G@     �E@      I@      D@     @R@     �X@     �Y@     �\@     �b@     �e@     �j@     `o@     @n@      p@     `r@     @o@     �l@      i@      g@      b@      ^@     �T@     �P@      N@     �H@      F@      :@      2@      0@      4@      (@      (@      ,@       @      "@      @      @      @      �?       @      �?      @      �?       @              �?               @              �?              �?        ��B3      �%��	��[���A*�f

mean squared error�,=

	r-squared`��=
�L
states*�L	   @���   ���@   �$[RA!�>��R��)J��妀A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             L�@     ș@     L�@     ��@     L�@     Π@     ��@     �@      �@     h�@     ±@     ?�@    ���@    ���@    �[�@    ���@     :�@    ��@    ��@    ���@    ���@     ��@    �V�@    ���@     b�@    �c�@    ���@    �	�@    �N�@    ���@    �K�@    `��@     ��@    ���@     
�@    �^�@    @��@    ���@     )�@    @��@     5�@    ���@     O�@    ���@    ���@    �5�@    @a�@    ���@    `��@    �a�@    �U�@    ���@    ���@    ��@    @��@     ��@    ``�@    ���@    ���@    @��@     !�@    ���@    ���@     <�@    ���@    ���@     S�@    ���@    �j�@    @��@     ��@    �V�@    @#�@    �0�@    �c�@     ��@     =�@     ��@    ���@    �j�@    �n�@    �K�@     ��@     ��@     ��@     ��@     ޻@     ��@     B�@     �@     ҵ@     ʹ@     5�@     %�@     ?�@     8�@     ��@     @�@     D�@     >�@     �@     X�@     ��@     `�@     ֦@     V�@     2�@     ,�@     ,�@     ��@     �@     ��@     ��@     .�@     ,�@     <�@     �@     ��@     \�@     ��@     ̜@     ��@     ��@     ��@     ܘ@     ��@     t�@     8�@     ��@     D�@     d�@     ��@     �@      �@     $�@     ��@     �@     ��@     t�@     `�@     |�@     ��@     <�@     �@     ԑ@     đ@     `�@     �@     \�@     `�@     ��@     (�@     X�@     ��@     ��@     �@     ��@     0�@     X�@     ؋@     P�@     X�@     p�@     �@     ��@     ��@     x�@     ��@     ȇ@     x�@     ؆@     h�@     ��@     @�@     ��@     @�@     ��@     ��@     ��@     (�@     ��@     ��@     ��@     ��@     ��@     P�@     �@     0�@     p@     X�@      �@     ��@     Ѐ@     P�@     ��@     �~@     0~@     ��@      �@     `~@     0}@     �{@      {@     �|@     �z@     P}@     �z@     �z@     �}@     pz@     0z@     �w@     �x@     �w@     �w@     �v@     px@     �u@     @u@     �u@     �s@     t@     �t@     �t@     `x@     �s@     �r@     �t@     �q@     s@     @r@     �q@     `o@      m@     �p@     �m@      o@     �n@      p@     �p@     �m@      n@     �j@     �m@     @o@     �j@      p@      k@     �p@      j@     `k@      m@      h@     �i@     @i@     �h@     �d@     �h@      h@      f@     �f@     @i@     �d@      g@     �e@     �c@     �d@     �e@     `d@     @c@     `a@      a@     �b@     �d@     �b@      a@     �c@      c@     �a@      b@      b@     @`@     �\@     �[@     @\@     @_@     @[@     �[@     �^@      X@     @Y@      ^@     �_@      \@      [@     @Z@     �W@     �]@      V@     �Y@     �Z@     �X@     C�@     1�@     �_@     �c@     �[@     �[@      \@      [@     �]@     �\@     �_@     �[@     �W@      \@     �^@     ``@     �`@     �d@     �f@     �`@      c@     �d@     �a@     `b@     @d@      b@     �`@     @a@      b@     `e@     `e@     �d@     �e@     �c@     `e@     �f@      g@     �g@     �c@      e@     �e@     @e@     �g@     @j@     �g@     `j@     `g@     �j@     `i@     @i@     �j@      i@      j@     @k@     @p@     @n@     `n@      n@      n@     Pp@      n@     �p@     �p@     p@     q@     �p@     �q@     �r@     �p@     �r@     �q@      s@     `q@     �r@     Pt@     0r@     �s@     �t@     @w@     �t@     �s@     �t@     �u@     �u@     `u@     �u@     `t@      v@     �u@     �y@     Pv@     �u@     �z@     p}@     �z@     @z@     �{@     �x@     P|@     �z@     0|@     ��@     `|@     `|@     �|@      ~@     �}@     p}@     �@     �@     �~@     �@     (�@     8�@     ��@     x�@     �@      �@     �@     p�@     ��@     ��@     Ѓ@     ��@     `�@     ��@     0�@     ��@     P�@     0�@     ��@     @�@     �@     ��@     ��@     �@     x�@     �@     �@     `�@     �@     ��@     p�@      �@      �@     ��@     Ȏ@     ȏ@     ��@     ��@     ��@     ��@     �@     ؐ@     �@     |�@     `�@     Ē@     �@     �@     8�@     ��@      �@     ��@     ��@     ��@     �@     ��@     |�@     ��@     З@     ��@     ؙ@     ��@     ��@     ��@     �@     ؙ@     ��@     x�@     ��@     ��@     П@     ��@     ؟@     ��@     J�@     ��@     `�@     ��@     <�@     z�@     ��@     n�@     <�@     ��@     ��@     ��@     .�@     6�@     �@     d�@     �@     .�@     ȭ@     z�@     ư@     r�@     ñ@     ��@     ��@     �@     ��@     I�@     �@     ޹@     V�@     ��@     ��@    ���@    ���@    ��@     �@     ��@    �o�@     Q�@     -�@    �L�@    ���@     ��@    �A�@    �y�@    @��@    ���@    �U�@    �L�@     '�@    �c�@    @��@     ��@     .�@    `��@    @��@    @P�@    ���@     w�@     ��@     �@     M�@    @��@    ���@    �M�@    @��@    @6�@    ���@    ��@    �)�@     ��@    @��@    �n�@     ��@    �!�@     |�@    ���@    `L�@    ���@    `��@    �~�@    @|�@    ���@    �E�@    �6�@    ���@     C�@    ���@     ��@    ���@     ��@    ���@    @��@    @��@    @l�@    �;�@     ��@    ��@    @U�@     �@    �d�@    ���@    @�@    �,�@    ���@     ��@     4�@     R�@     v�@     ��@     P�@     �@     ��@     x�@     x�@     Ȃ@     ��@        
�
predictions*�	   @��    �@     ί@!  nj�A@)�����R.@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲��ѩ�-߾E��a�Wܾ8K�ߝ�>�h���`�>�ߊ4F��>��Zr[v�>O�ʗ��>�FF�G ?��[�?6�]��?����?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?��tM@w`<f@�������:�              �?      �?      @       @       @      @      &@      0@      4@      4@      >@      B@     �A@      I@      L@      M@     �Q@     @Q@     �P@     @T@     @P@     �K@     �J@      U@     @P@     �P@      I@      H@     @P@      L@      J@      F@     �D@     �C@      8@      B@     �@@      @@      :@      ;@     �@@      2@      6@      2@      (@      2@      *@      &@      *@      @      @      @      @      @      @      @      @      @      @      @      @      @      @      �?       @       @      �?              �?      @      @      �?      �?              �?       @      �?      �?      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?       @              �?               @              @              @      �?               @      @      @       @       @      @      @              @       @      "@      @       @       @      @      @      @      1@      "@      3@      ,@      9@      7@      9@      6@      7@      9@      5@      ?@      ?@      G@     �F@      C@      L@     �O@     @Q@     �L@      S@      Q@     @S@     �S@      T@     �T@     @R@     �M@     �T@     �S@      M@      I@      L@      K@     �J@      M@      F@      B@     �G@      8@      ?@      2@      ,@      2@      3@      0@      "@      ,@      (@       @      @      @       @      @       @      @      @      @      @       @       @      �?       @      �?      �?               @      @              �?      �?              �?              �?        ���B2      ��}�	��l\���A*�d

mean squared error�9*=

	r-squaredt7>
�L
states*�L	    ���   ���@   �$[RA! j�@s��):m�
��A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             P�@     ��@     $�@     8�@     L�@     ��@     ܡ@     ��@     D�@     t�@     8�@     }�@    `��@     ��@    �S�@    ���@    ���@     �@     P�@      �@    ���@    @�@    ���@     [�@    ���@     o�@     ��@     Y�@     b�@    �=�@    ���@    �}�@    `c�@    @��@    ���@    ���@    @��@    ���@     ]�@     /�@    @��@    ���@    ��@    ���@    `��@    ��@    ���@    ���@    �7�@     ��@     _�@    ���@    ���@    `x�@     �@    ���@    ���@    �,�@     [�@    ���@     ��@    ��@    �w�@     }�@    ���@    �;�@    @��@     ��@    ���@    @:�@    ���@    �z�@     >�@    ���@     f�@    ��@    ���@    ���@     ��@    ���@    ���@     ��@     �@     ��@     û@     ��@     ��@     ��@     ��@     �@     �@     ��@     q�@     ��@     ��@     p�@     ֬@     �@     R�@     Z�@     (�@     ��@     ��@     T�@     �@     ޣ@     ��@     ��@     <�@     ơ@     �@     ��@      @     П@     $�@     �@     ��@     ��@     ȝ@     \�@     p�@     t�@     0�@     ��@     ��@     4�@      �@     @�@     ��@     ��@      �@     |�@     �@     ��@     ��@     ��@     �@     <�@     ��@     ��@     ��@     \�@     ؐ@     ��@     L�@     ��@     ��@     `�@     ��@     �@     p�@     ȏ@     ��@     ��@     ��@     �@      �@     8�@     ��@      �@     P�@     X�@     ��@     �@     8�@     ��@     @�@     p�@     X�@     ��@     ��@     �@     8�@     H�@     8�@     `�@     (�@     0�@     Ѓ@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     H�@     ȁ@     ��@     �@     8�@     (�@     ؀@     �{@     @}@     �|@     �~@     0{@     �z@     @z@     �{@     �{@      |@     �}@      {@     �|@     P{@     �z@     0x@     z@     �{@     �w@     �x@     �y@     �y@     0x@     @w@     �v@     0x@     �w@     u@     Pt@     0v@     �u@     Pt@     �r@     Ps@     @r@     0r@     �s@     �q@     �q@     t@      q@     �q@     �n@     �p@     �o@     �p@     r@     �r@     �o@     �m@     �l@      m@     �q@      k@      k@     `k@      h@     �h@     `k@      i@      j@     @e@     �g@     �d@     �g@      e@      e@     �e@     �e@     @e@     �d@     �e@     @d@     �b@     @e@     `c@     @b@     �_@      d@      d@     `f@      a@      `@     @c@      b@     �`@     �c@      d@      b@     �`@     �\@      a@     @Z@      \@     �b@      \@     �\@     �`@     @\@     @`@      Y@      \@      a@     @U@     �X@     �X@     �Y@     �V@     ?�@     8�@      \@     �]@     �`@     �`@      \@      a@     �b@      c@      b@     �]@      ^@      ^@     �^@     �a@      a@     `c@     @c@     �a@     �d@     �d@     `a@     �b@     @b@     `c@     �f@     �d@      d@     �d@      e@     �f@      g@     �d@     @h@      f@     �g@     @g@     �e@     �f@     �g@     �k@     `h@     �j@      k@      j@     �k@     `j@     @n@     @n@     �l@     �m@     `m@      o@      m@     �m@     �p@     �n@     �p@     �p@     @o@     @o@     �o@      r@     @r@     0r@      s@     r@     �s@     �s@     �q@      t@     �q@      t@     �r@     t@     �w@      v@     �w@     Pz@     w@     �u@     �x@      w@     �u@     �v@     @w@     �y@     �v@     �y@     p~@     �y@     @y@     {@     �x@     P{@     �{@     �{@     �|@     �{@     `y@     �|@     �|@     �~@     @@     �@     �@     X�@     H�@     @�@     `�@     @@     ؀@     X�@     0�@     h�@     ��@     (�@     �@     ��@     �@      �@     ��@     ؄@      �@     ��@     (�@     x�@     Ї@      �@     �@     �@     ��@     8�@     (�@     0�@     p�@     ��@     @�@     (�@     0�@     H�@     ��@     ��@     �@     h�@     ��@      �@     ��@     ��@     `�@     x�@     Џ@     đ@     L�@     ��@     В@     x�@     ,�@     ԓ@     ��@     t�@     ��@     �@     `�@     ��@     ��@     ؖ@     �@     ȕ@     d�@     ��@     |�@     \�@     ��@     �@     H�@     ��@     ��@     ��@     ؛@     (�@     �@     �@     ��@     ؞@     Ğ@     \�@     ֢@     Z�@     ~�@     6�@     ��@     �@     Ĥ@     �@     P�@     ��@     ��@     j�@     Z�@     H�@     ��@     H�@     ��@     �@     �@     r�@     *�@     ��@     ��@     [�@     ��@     �@     ��@     ��@     ȷ@     s�@     ��@     �@     {�@    �g�@    ���@    �q�@     ��@     *�@    ���@     u�@    ���@     ��@    �m�@    ���@    @�@     ��@     '�@    @��@     v�@    �O�@    ���@     j�@     �@    �1�@     ��@     L�@     ��@    ���@    �K�@    `��@     ��@    ���@    @Y�@    `��@     ��@    �v�@    ��@    @��@    �b�@     �@    ��@    ��@    `��@    �!�@    `?�@    ���@    ���@    ���@     [�@    ���@     ��@     ;�@    ���@    `��@    ���@    @y�@    �8�@    �t�@    �c�@     n�@     )�@     ��@    ���@     `�@    ���@    @��@    @��@    @��@     ��@    � �@    ���@    ���@     ��@    ` �@     Q�@     ��@     �@     ��@     ��@     P�@     ,�@     ܐ@     ��@     ��@     `�@     l�@        
�
predictions*�	   �K긿   `�9@     ί@!  �w�C�) ޾=�^1@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�6�]���1��a˲�['�?�;;�"�qʾ��n����>�u`P+d�>����?f�ʜ�7
?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?ܔ�.�u�?��tM@w`<f@�6v��@�������:�              �?      @      @      &@      7@      B@      F@     @Q@     �Q@     �X@      a@     �`@      d@     @f@     `f@      h@     �g@     `g@     �g@     �d@      c@     �_@      Z@     @Y@     �S@     �W@     @S@     �P@     �G@      L@      A@     �F@      7@      8@      7@      7@      (@      ,@      "@      $@      @      (@      "@      @      @      @      @      $@      @       @      @      @       @      �?      @       @       @      @       @      @              @      �?              �?              �?      �?      �?      �?              �?       @              �?      @      �?              �?              �?              �?              �?              �?              �?      �?      �?              �?               @      �?       @      @               @      @      @       @      @       @      @       @      @      @      @      @      @      @      @      @      0@      @       @      @      @      @      "@       @      "@      *@      0@      .@      @      ,@      ,@      (@      1@      4@      6@      <@      2@      3@      7@      5@      ,@      @@      <@      4@      1@      7@      5@      ,@      5@      2@      1@      0@      (@      *@      @      (@      *@      @      @      @      @      @      �?      @       @      @              @      �?      @              @              �?      �?              �?               @              �?              �?        �T��3      _  	���\���A*�g

mean squared errorB�/=

	r-squared���=
�L
states*�L	    ���    ��@   �$[RA!q�R����)R��j�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ؙ@     ��@     Ȑ@     H�@     ��@     (�@     >�@     ��@     h�@     ,�@    ���@     ��@     ��@    @��@    �z�@    ��@    ���@    ���@    �Q�@    @H�@    @d�@    �	�@    ���@    �C�@    ���@    @��@    �f�@    @��@     ?�@     ��@     (�@    �e�@    @��@    ��@    ���@    �J�@    �n�@    @��@    �/�@    ���@     ��@     "�@    ���@    �'�@     5�@    `O�@    `T�@    �3�@    ��@    `��@    `7�@     ��@    �'�@     r�@    `��@    �q�@    @��@    �n�@    @��@    �C�@    �(�@    @p�@    �p�@     
�@    �\�@     ;�@    ���@    ���@    �7�@    ��@     E�@    ���@     �@     ��@    ���@      �@    �K�@     q�@    �t�@     ��@    ���@     ��@     ��@     ־@     ��@     ��@     �@     ��@     !�@     	�@     ϴ@     ��@     ��@     }�@     =�@     J�@     ��@     F�@     �@     H�@     �@     .�@     h�@     �@     ��@     �@     ��@     "�@     ��@     ��@     ��@     �@     8�@     �@     4�@     ��@     ��@     l�@     ��@     X�@     ��@     H�@     ��@     Ș@     $�@     P�@     �@     \�@     �@     Ȕ@     ��@     D�@     ԓ@     Ĕ@     x�@     �@     ��@     l�@     ̒@     H�@     x�@     ��@     ��@     ؐ@     $�@     ؎@     h�@     8�@     p�@     �@     �@     `�@     �@     ��@     p�@     ؍@     ��@     ؋@     ؊@     h�@      �@     ��@     ��@     ��@     8�@     ��@     ��@     ��@     X�@      �@     ��@     ؅@     �@     ��@     ��@     ��@     ��@     8�@     ��@     ��@     `�@     ��@     �@     Ђ@     `�@     0�@     Ђ@     �@     ��@     P�@     �@     h�@     p|@     p~@     �@     0|@     �~@     �|@      @      }@     �|@     �y@      |@     �y@     �z@     py@     @y@      y@     py@     �z@     �z@     �u@     px@      x@      v@     �u@     �x@     `v@     �t@     0w@     Pu@     z@     @w@     0u@     �w@     �u@     �u@     s@     �u@     pu@     Pu@     @q@     �r@      t@     �q@     pq@     0p@     �q@     @m@      p@     0p@     0p@      q@     �q@     �j@     �m@     �l@     �q@     �o@     �i@     �l@     �l@     `p@     `k@     �j@     �f@      f@     @j@     �g@     @g@     �b@     `f@     �e@     �h@      h@     �e@      g@      d@     @d@     �c@      a@     �f@      c@     `b@     @f@     �f@      c@     @d@      e@     �d@      c@     �`@      `@      a@     �`@     �b@      b@      b@     @\@     �`@      ^@     @a@      [@     @a@     �\@     @\@     �\@     @Y@     �^@     �[@      `@     ��@    �$�@     �a@     �`@      `@     �^@     �`@     @`@      \@     �a@     �]@     �Y@      b@     �a@     �b@     @_@      `@      d@     �d@      e@      e@     �d@     @f@     @h@      g@     �c@      c@     `e@      h@     �h@     �d@      i@     @i@     �h@      h@     �h@     �g@     @h@      f@     �h@     `j@     @h@     �j@     �j@      g@      j@     �l@     �k@     �k@     �n@      p@     �q@     @p@     �p@      q@     �m@     �q@     �p@     �q@     �p@     @p@      q@     �q@     q@     �s@     �s@     �q@     �r@     �r@     Ps@     @r@     u@     �r@     �r@     ps@      u@     0u@     @v@     @y@     �u@      v@      w@     0y@     @y@      y@     @y@     px@     @{@     �w@     {@     {@     �~@     �|@     p{@     �{@     @y@     p|@     �{@     `{@     @{@     �~@     �~@     �@     `@     �@     P�@     Ё@     ��@     ��@     ��@     �@     @�@     ��@     �@     Ђ@     ��@     �@     �@     h�@     �@     �@     �@     ��@     ��@     p�@     ��@     @�@     ��@     ��@     �@     Ȉ@     �@     ��@     ��@     ��@     p�@     (�@     ��@     ��@     X�@      �@     ��@      �@     P�@     ��@     ��@     X�@     �@     �@     |�@     ؐ@     t�@     ��@     ��@     �@     ��@     X�@     T�@     l�@     @�@     ��@     ��@     p�@     ��@     ��@     ��@     ��@     x�@     Е@     ��@     ,�@     Ԙ@     �@     ��@     ܙ@     ��@     0�@     D�@     �@     �@     ��@     ��@     ��@     8�@     ��@     ��@     �@     �@     0�@     x�@     ��@     ��@     ��@     �@     <�@     h�@     ��@     �@     j�@     ��@     x�@     ^�@     *�@     T�@     �@     "�@     n�@     m�@     K�@     ��@     ��@     I�@     :�@     �@     ��@     ��@     �@     j�@     9�@     ��@     ��@     9�@    ���@    �\�@     �@    �y�@     ��@     T�@    ���@    � �@    �C�@    �p�@    �n�@    @��@    @��@    �t�@     U�@    �{�@    �.�@    ���@    @#�@     ��@    �I�@    `�@    @��@    `.�@    ���@    @0�@    ���@     ��@    @��@    `%�@    ���@    � �@    �
�@    ``�@    �s�@     |�@    @��@     R�@    @��@    @�@    @��@    `��@    �v�@     ��@    ���@    �S�@    `!�@    ��@    `��@    `��@    @��@     B�@    ���@    @��@    ���@    ���@    @��@    ���@     j�@    �Z�@     e�@    ���@     ��@     ��@    @@�@    @��@    @��@    ���@     I�@    ���@     ^�@     >�@     V�@     ڣ@     �@      �@     8�@     ��@     ��@     (�@     ��@     �@        
�
predictions*�	   �����   �1� @     ί@!  ��kT@)z.��9@2�� l(����{ �ǳ������iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�6�]���1��a˲���[���FF�G �>�?�s����h���`�8K�ߝ�u��6
�>T�L<�>�*��ڽ>�[�=�k�>K+�E���>jqs&\��>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?ܔ�.�u�?��tM@�������:�              �?              �?      @      @      @      &@      .@      1@      1@      C@     �C@     �E@      F@      M@      N@     �F@     �M@     �R@     �P@      L@     �O@     �J@     �H@      <@     �A@      H@      @@      B@      ?@      A@      ;@      5@      8@      <@      ;@      5@      1@      1@      "@      (@      $@      ,@       @      @      *@      "@      "@       @      @      @      @      @      @      @      @      @       @       @      @      @      @              �?      �?      �?      �?      �?      �?       @      �?              @      �?              �?      @              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?       @              �?      �?      �?      �?      �?      �?      �?       @       @      �?      �?       @      �?       @      @      @      @      @      �?      @      �?      @       @      @       @       @      "@      @       @      @      @      &@      2@      &@      3@      (@      >@      .@      @@      6@     �@@      >@     �B@      A@     �C@     �C@     �P@      P@     �H@      N@     �Q@      U@     �O@     �W@      U@      T@      V@     @W@      Y@      T@     �M@     @Q@      M@     @Q@      O@     �K@      P@      N@     �O@      Q@     �H@     �K@     �G@     �E@     �C@     �D@      ;@      *@      6@      ,@      1@      $@      $@      @       @      $@      @      @      @      @      @       @      @       @              �?      �?      �?      �?               @      �?              �?               @              �?        W�d�23      ��	��V]���A*�f

mean squared error��'=

	r-squared��>
�L
states*�L	   ����    ��@   �$[RA!+>����)��X)g�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             0�@     |�@      �@     P�@     ��@     v�@     �@     v�@     t�@     e�@     g�@     �@     ��@    �x�@    ���@     ��@     �@    ���@    ���@     ��@    ��@    ���@    @�@     Q�@    ���@    @-�@    ���@    ���@    ���@    ���@    �o�@     ��@    �=�@    `O�@    `��@     Q�@     k�@    `��@    @��@    `N�@     ��@    �x�@    �"�@    `x�@    @��@    ���@    @��@    �V�@    `��@    @��@    `.�@    ���@    �e�@     ��@    �T�@    `��@    �'�@    ���@    ���@    ���@    @��@     ��@    @��@    @=�@     ��@    @��@    ���@    @��@     v�@     ��@    ���@    ���@     ��@     ��@    ��@     ��@     �@     F�@     ��@     [�@     ��@     8�@     e�@     ��@     �@     G�@     �@     [�@     ��@     �@     �@     ��@     ��@     �@     x�@     װ@     �@     �@     @�@     ,�@     @�@     z�@     �@     ��@     �@     ��@     ��@     `�@     L�@     ��@     ��@     ��@     �@     ��@     ��@     ��@     ��@     ��@     Ȟ@     �@     ��@     ؛@     ��@     8�@      �@     L�@     ��@     |�@     L�@     p�@     ��@     P�@     �@     0�@     8�@     ��@     t�@     �@     ��@     ԑ@     ��@      �@     ܐ@     ��@     `�@     А@     0�@     D�@     \�@     ��@     ��@     ��@     ��@     ��@     p�@     8�@     X�@     p�@     ؎@      �@     @�@     Ћ@     p�@     p�@     �@     ��@     `�@     ��@     ��@     (�@     H�@      �@     ��@     (�@     P�@     ��@     ��@     ��@     Ȅ@     H�@     h�@     H�@     ��@     `�@     `�@     `�@     �@     ��@     ��@     h�@     0�@     ȁ@     x�@     8�@     h�@     �@     �}@     �}@     �}@     0~@     P|@      z@      {@     0{@     |@     0{@     �x@     �z@      z@     �w@      x@     �w@     �x@     �y@      y@      v@     �v@     Pv@     @v@     `u@      x@     pv@     �u@     �u@     �t@     �s@      w@     �u@     �w@     �q@      u@     Pu@     �r@     �r@     �r@     �q@      u@     �u@     �q@      q@     �p@     pp@      n@     �p@     �n@     �q@     Pp@     �m@      n@     �m@      n@      l@      n@     �p@     �g@      l@     �o@      m@      i@     �o@     �j@      k@     `n@      j@     �i@     @f@     �c@     �i@     �c@     �h@     �g@     �g@     �c@     �e@      e@     �g@     �f@      i@     �e@      d@     ``@     �`@     �c@     �e@     `a@     �`@     �a@     @`@      c@     �`@      a@     �_@     �_@     �`@     �`@     �c@     �^@      _@     @\@     �^@     @]@     �]@      �@    �w�@     @`@     @b@     �`@      d@      c@      a@     �_@     �a@     �d@     `c@     �`@     �a@     �b@      e@     �f@     �e@     �f@      d@      g@     �f@     @f@     `f@      f@     �g@     `e@     `g@     �h@     �f@     �i@      g@     �k@      j@     �h@     @j@      k@      e@     `k@     �k@     �j@      k@      l@     �h@     @i@     `k@     �n@     �n@     �n@     @m@     @m@     �l@     �o@     �q@     �k@     @q@      s@     �m@     q@     0q@      p@     �q@     Pr@     `s@     0t@     �t@     �r@     `s@     �u@     �t@     �t@     �t@     pu@     �u@     v@     �z@      y@     0w@     �y@     �x@     �w@     �x@     pz@     �z@     `w@     �y@     �v@     �v@     P{@     �{@      {@     `~@     �x@     �{@      z@     Pz@      |@     @~@     @z@     �}@     �|@     �|@     �@     �@     @�@     ��@     ��@     ��@     Ё@     h�@     ��@     �@     Ѓ@     ��@     ��@     ��@     ��@     ��@     ��@     H�@     ��@     `�@     �@     ��@     ��@     ��@     �@     8�@      �@     І@     ��@     ��@     x�@     ��@     �@     Ћ@      �@     X�@     ��@     ��@     ��@     ��@     X�@     ��@     ��@     ��@     ؎@     ��@     ��@     ؐ@     ��@     h�@     �@     X�@     ��@     ��@     X�@     x�@     �@     `�@     ��@     ��@     ��@     Ԕ@     ��@     H�@      �@     ��@     L�@     �@     ė@     T�@     ,�@     �@      �@     ��@     �@     �@     D�@     ��@     ��@     ��@     ��@     p�@     �@     T�@     ��@     Ρ@     ��@     j�@     P�@     آ@     ޢ@     |�@     أ@     �@     ��@     �@     �@     ��@     ��@     ��@     ��@     B�@     ��@     ޮ@     )�@     �@     ��@     �@     �@     ��@     &�@     �@     -�@     ��@     �@     i�@     {�@     ��@     D�@    ���@     ��@     6�@    ���@     $�@     ��@     ��@    �e�@     �@    �]�@    @Q�@    �-�@     ?�@    @>�@     ��@     ��@    ���@    ���@    @��@     *�@    @��@     ��@     ��@    ���@    �e�@    ���@    �j�@    ���@    ���@     �@    �q�@    @��@    @>�@     ��@     ��@    ���@    ���@     ��@    ���@     C�@    �J�@    ���@    ���@    �~�@    ��@    ���@    @��@     ^�@    `/�@     ��@    ���@    ���@    ��@    `k�@     ��@     %�@     ��@    ���@     0�@    @��@    @��@    �f�@    �&�@    ��@     �@    ���@    @��@     ��@    �+�@     (�@     T�@     b�@     �@     ��@     b�@     ��@     l�@     �@     ��@      �@     Ѕ@     (�@     ��@        
�
predictions*�	   �1X��   �G?@     ί@!  �+�@)H)_���1@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
������6�]���1��a˲���[���FF�G �>�?�s���O�ʗ�����Zr[v��I��P=��pz�w�7��8K�ߝ�>�h���`�>1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?�E̟���?yL�����?h�5�@�Š)U	@�������:�              �?      �?       @      @      @       @      .@      8@      ?@      F@      L@     �J@     @Q@      Q@      Q@      S@     @Y@      X@     �V@     @Z@      X@     @W@     @V@     �U@      S@     �R@     �R@     @Q@     �Q@      Q@     �B@     �B@      G@     �C@     �I@     �B@      E@      9@      =@      @@      .@      :@      .@      0@      "@      2@      @      $@      &@      &@       @      @      @       @      @      @       @      @      @      @      �?      @      @      @      @      @              �?      �?      �?              �?              �?      @              �?               @      �?      �?              �?      �?      �?       @              �?              �?              �?              �?              �?      �?      �?       @      �?      �?              @              @       @      @              �?       @      �?       @       @       @      @      @      @      @      @      @      @      "@      @      "@      *@       @      $@      ,@      ,@      "@      2@      &@      =@      2@      5@      =@      >@      ;@      <@      2@      E@     �G@     �A@      A@      G@     �@@     �G@      I@     �L@      O@      N@     �P@      G@     �I@     �F@      G@     �H@      K@      E@     �G@     �G@      H@     �D@     �D@      >@      @@     �A@      .@      7@      0@      @      &@      "@      @      "@      @      @               @       @       @      �?      @      �?              �?      @              �?              �?              �?              �?              �?        �.�
�2      �	���]���A*�e

mean squared error#�)=

	r-squaredDt
>
�L
states*�L	   ����   @��@   �$[RA!��Ow�u��)2�j�F�A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     4�@     ��@     �@     ��@     h�@     2�@     �@     �@     ��@     �@    ���@    �s�@    ��@    �1�@     j�@    �t�@    �4�@     8�@    �g�@    @��@     ��@    �P�@    ���@    @�@    ���@    ��@    ���@    �{�@    ���@    �t�@     y�@    ���@    ���@     ��@    ���@     ��@    �0�@    `��@    �/�@     ��@     ��@    �5�@    ��@     �@    ��@    ���@    @T�@     =�@     ��@     X�@    �6�@    �`�@    ���@    �U�@     ��@    ��@    ���@    ���@    �\�@    ���@     ��@    �U�@    �#�@    �Y�@    �(�@    �`�@    �A�@    �V�@    @}�@     ��@     }�@    ���@     ��@     &�@    �T�@     ��@    ���@     #�@     ��@    ��@     '�@    �	�@    �k�@     ��@     	�@     ��@     0�@     η@     ��@     /�@     ��@     ��@     �@     K�@     t�@     �@     �@     .�@     &�@     v�@     ��@     >�@     ��@     ��@     
�@     P�@     .�@     v�@     �@     T�@     d�@     �@     ܠ@     ��@     :�@     ��@     ��@     ܜ@     ��@     �@     ��@     \�@      �@     �@     ��@     �@     ��@     ��@     ��@     ��@     ��@     p�@     ��@     �@     (�@     �@     �@     ,�@     \�@     Ȓ@     ��@     ��@     ��@     ,�@     ,�@     $�@     �@     0�@     x�@     ��@      �@     ��@     8�@     ��@     ��@     (�@     �@     ��@     ��@     ��@     Њ@     ��@     x�@      �@     �@     �@     ؆@     ��@     ��@     ��@     ��@     p�@      �@     X�@     h�@     ��@      �@     x�@     8�@     ��@     ��@     ��@     H�@     �@     �@     ��@     ��@     0�@     P�@     H�@     @�@     ��@     p�@     �}@     @     p@     `@     �}@     �|@      ~@     �y@     �z@     �{@     �|@     �y@     `{@      ~@     �|@      z@     �x@     0v@     �u@     �v@     Pz@     �x@     �v@     @w@     x@     pu@     w@     �v@     0u@     �u@      u@     �t@     �v@      u@     `v@     �v@     Pt@     `r@     �u@     pq@     �s@     �s@      r@     �p@     �q@     @q@     `s@     0r@     �p@      p@     pp@     Pp@     �m@      p@     �p@     @p@     �m@     �p@     �q@     0p@     �m@     �g@      k@     `p@     �l@     �j@      g@     �i@     �h@     �i@      j@      l@     �l@     �i@      i@      o@     `k@     `n@     `h@     �g@      f@      k@     �f@     �c@     @e@     �d@     `b@      d@      b@     `g@      b@     �b@     @b@      `@      a@     @b@      a@      a@     �a@     `a@     `d@     @_@      `@     �_@      _@     �_@     �`@    ���@    ���@      a@     @b@     �`@      a@     @`@     �b@     �b@     @c@      b@     �d@     @a@     @i@     �f@     �e@      g@     �d@     �f@     �i@      g@     `g@      e@     �g@      g@     �h@     �g@     �g@      l@     �f@      i@      h@     �f@     �k@     �j@     �j@     `k@     �m@      l@     @l@     �j@     @m@     �n@      o@     �m@     �n@     �k@     @p@     Pp@     Pp@     �o@     �p@     �q@     �r@     0r@     @q@     �p@      r@     �p@     �t@      u@     @r@     �s@     `s@     @q@      s@     ps@     Pt@     0t@     0u@     �s@     �t@     Pv@     x@     �v@     `w@     �v@     py@     0w@     �v@      w@     `y@     @v@     �v@     x@     @x@     pz@     �x@     �x@     py@     0}@     @{@     �z@     `@     `|@     `|@     0y@     @�@     0@     �@     �@     �@     x�@     (�@     ��@     ��@      �@     X�@     �@     p�@     ��@     P�@     H�@     (�@     �@     Є@     @�@     ؃@     ��@     ��@      �@     X�@      �@     ��@     ��@     �@     X�@     ��@     0�@     ؇@      �@     �@     ��@     ��@     �@     ��@     ��@     ��@     �@     ��@     @�@     ��@     ��@      �@     @�@     X�@     �@     p�@     4�@     \�@     8�@     �@      �@     ��@     �@     P�@     �@     ��@     ��@     L�@     �@     P�@     �@     h�@     $�@     l�@     Ȗ@     �@     `�@     Ж@     D�@     ȗ@     ��@     ̚@     �@     ��@     �@     ̚@     Н@     �@     0�@     $�@     ��@     Ğ@     Ȟ@     �@     
�@     ��@     ��@     F�@     ��@     �@     R�@     ��@     T�@     ��@     ̥@     T�@     |�@     �@     �@     ��@     ��@     @�@     .�@     ��@     3�@     e�@     ��@     ��@     u�@     �@     �@     t�@     O�@     ��@     ��@     F�@     �@     ��@      �@     ��@     ��@     b�@    ���@     ��@     ��@    ���@    ���@    @�@    @�@    �P�@     �@    ���@    ���@    @x�@    �6�@    �:�@     +�@     X�@    @b�@     B�@    ���@     ��@    ���@    �l�@     b�@    @��@     ��@     ��@    ���@     %�@    �G�@    ���@    `��@    �J�@    �K�@    ���@    `��@     ��@    ��@    ���@     �@    ���@    @;�@    ��@     ��@    @��@     J�@    `�@    ���@    @��@    @��@     ��@    �u�@     -�@     ��@    ���@    ���@    @9�@    @d�@    �s�@     r�@    ���@    @��@    �8�@    ���@    ���@    ���@     ��@     T�@    @��@     ξ@     �@     $�@     �@     �@     <�@     ��@     0�@     �@     �@     h�@     ĕ@        
�
predictions*�	   @�ɿ   `�4@     ί@!  ��w�G�)���h��3@2��K?̿�@�"�ɿ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
��FF�G �>�?�s������%ᾙѩ�-߾a�Ϭ(�>8K�ߝ�>f�ʜ�7
?>h�'�?�T7��?�vV�R9?�[^:��"?U�4@@�$?�7Kaa+?��VlQ.?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?2g�G�A�?������?3?��|�?�E̟���?h�5�@�Š)U	@�������:�              �?              @      @      @      3@      1@      E@     �M@     �S@      O@     �V@     @U@     @Z@      [@      _@      `@     @]@     �^@     �^@     �_@     �_@      Y@      ]@     �\@     �R@     @V@     @R@     �R@      R@     �I@     �L@     �K@      G@      F@     �D@      E@      >@      ?@      ,@      3@      .@      1@      7@      0@      *@      "@      &@      @       @       @      "@      @      $@      @       @      "@      @      @      @       @      @      �?      @       @       @      �?      �?      �?      @      @      �?       @              �?       @      �?      �?               @      �?              �?              �?      �?              �?              �?              �?              �?               @               @              �?               @       @      @       @       @               @      �?              @      @      @       @      @      @      �?      @      @      $@      @      @      *@      (@      &@      0@      $@      1@      ,@      ;@      2@      7@      9@      >@      6@      A@      0@      E@     �D@      @@     �@@     �@@     �C@      <@      <@      @@     �D@      B@     �A@      G@      <@      8@      @@      @@      3@      <@      5@      5@      1@      0@      ,@      @       @       @      @      @      @      @       @      @      @      �?              �?      @       @       @      �?       @      �?       @              �?              �?              �?        dƪ�2      �	�hQ^���A*�e

mean squared errorA�0=

	r-squaredp��=
�L
states*�L	    ���   ���@   �$[RA!��=�R��)�$�SWA2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Z�@     `�@     x�@     $�@     \�@     &�@     ��@     �@     U�@     ��@     ��@    �p�@     D�@    ���@    ���@     ��@    �"�@    ���@    �T�@    ��@     ��@    ���@     ��@    ��@    @0�@    ���@    @C�@    �O�@    @��@    @T�@    ���@    @��@    @��@     ��@    �-�@    `/�@    �,�@    `��@    @�@     ��@    @/�@     ��@    ���@    ��@    ��@    ��@    @��@    �j�@    �Q�@     V�@     ��@     ��@    �L�@    `��@    ��@    `��@    @�@    �"�@     ��@     ��@    ���@    �s�@    @�@    ���@    ���@    ���@    ���@    ���@    �M�@    @[�@    ��@    �`�@    ���@     ��@    ��@     ��@    �S�@     ��@     b�@    ��@     ��@     )�@     �@    ��@    ��@     ��@     o�@     �@     q�@     ��@     �@     R�@     ��@     ��@     г@     0�@     ��@     ɰ@     ֯@     Į@     ��@     ެ@     ڪ@     B�@     8�@     ��@     ��@     �@     ��@     X�@     B�@     ��@     �@     f�@     �@     &�@     >�@     �@     D�@     Ğ@     l�@     H�@     Ԝ@     ��@     ��@     ��@     T�@     ,�@     ��@      �@     �@     Е@     t�@     ��@     ��@     p�@     @�@     \�@     l�@     ��@     p�@     t�@     �@     ��@     Б@     Ԑ@     \�@     ��@     ��@     ��@     @�@     ��@     ��@     x�@     0�@     @�@     ��@     Ѝ@     �@     ��@     ��@     @�@     ��@     ��@     ��@     p�@     ��@     ��@     ��@      �@     x�@     X�@     ��@     �@     (�@     8�@     �@     ��@     X�@     ȃ@     p�@     ��@     ��@     ��@     x�@     8�@     �@     ��@     Ȃ@     ��@     ��@     ��@     `�@     ��@     `@     ��@     H�@     �~@     �@     �{@      |@     P@     �{@     p}@     {@     �~@     �~@     p}@     �|@     �w@     �y@     y@     @x@     �{@     �x@     �x@     �z@      y@     x@     0v@     �w@     0w@     �u@     pt@     �s@     �t@     ps@     �v@     s@     `t@     0t@     �r@     �t@     �s@     @u@     �t@     0u@     0u@     �p@     0r@     �p@     �p@     `r@     �r@     Pp@      q@     pp@     �p@      n@     p@     @n@      o@     �p@      n@     `k@      l@      n@      i@      m@     @n@     �o@      n@     @i@      l@     pp@      j@     �j@     `i@     �g@     @i@     @h@     `f@      k@      j@     �h@      e@     `g@     `f@     @i@     �j@     �f@     �f@     �b@      g@      d@      c@     �c@     �`@      a@     �e@     �g@      g@     `c@      `@     �`@     �a@     �a@     �`@     �^@     �`@     `a@     �]@     ��@     `�@      a@      b@      a@     @`@     �c@     �e@     `a@     `b@      e@     �e@     �c@     `g@     �b@     `c@     `e@     �d@     �g@     �f@     @f@     `d@     �e@     �h@     `h@      f@     �f@     @f@     �i@     `g@      i@     �e@     @j@      i@     @k@     �j@     @m@     �j@     `m@     @l@     `k@      n@     �o@     `n@     �n@     �o@      m@      t@     �o@     `o@     pp@     0p@     pp@     Pr@     �p@     `r@     0u@     ps@     �p@     Ps@     �r@     `q@     Pt@     �q@     0q@     �r@     pq@     �t@      t@     u@      u@     �v@     �u@     �w@     @y@     �t@      v@     �w@     �z@     �y@     �u@     �v@      w@     w@     �z@     �y@     `{@     {@     �y@     0z@      {@     @|@     P{@     �|@     �|@     �@     �@     @     �{@     (�@     0�@     x�@     8�@     x�@     ��@     0�@     Ȁ@     ��@     ��@     ��@     0�@     ��@     �@     ��@     �@     X�@     ��@     ��@     �@     ȅ@      �@     �@     ��@     І@     �@      �@     ��@     ��@     0�@     ��@     �@     ؈@     ��@     ��@     �@      �@     Њ@     8�@     `�@     p�@     H�@     P�@     �@     ��@     ��@     ؍@     (�@     h�@     �@     Б@     `�@     L�@     ,�@     ��@     �@     ̑@     ̓@     ��@     ��@     ��@     x�@     ��@     ��@     h�@     (�@     d�@     �@     ��@     ��@     ��@     З@     ș@     Ț@     \�@     �@     D�@     ��@     �@     ��@     <�@     ��@     L�@     О@     ��@     П@     L�@     d�@     ��@     Т@     "�@     0�@     ��@     ��@     v�@     ��@     ¥@     D�@     �@     8�@     $�@     p�@     ��@     ~�@     ޮ@     ��@     ��@     �@     ��@     4�@     ��@     :�@     Զ@     ��@     ��@     ��@     ��@     �@    �&�@     �@    ���@     I�@    ���@     ��@    �F�@    ���@    �w�@    ���@     a�@    �	�@     Y�@     H�@    ��@     E�@    ��@    �U�@    �2�@    ���@     ��@    ���@    �c�@    ���@     N�@    �q�@     ��@     K�@    ���@     O�@    ���@    �u�@    @V�@    ���@     a�@    @��@     *�@    `T�@    �{�@    `��@    �r�@    �%�@    ���@    �d�@    @��@    @X�@    ��@    ���@    ���@    ``�@     A�@    ���@    ���@    �w�@    @��@     �@    ���@    �L�@     �@    �G�@    @p�@    ���@    �k�@     "�@    @L�@     ��@    @��@     ��@     w�@    ���@     ��@    ���@    ���@    �"�@    `!�@     ��@     w�@     �@     �@     `�@     L�@     (�@     0�@      �@     ��@     ��@     ��@        
�
predictions*�	   �_I��    �@     ί@!  ���R@):&@��d>@2���]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'�������6�]�����(��澢f����;�"�qʾ
�/eq
Ⱦ�ѩ�-�>���%�>�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?S�Fi��?ܔ�.�u�?u�rʭ�@�DK��@�������:�               @      @      @      1@      "@      8@      ?@      ?@      J@     �C@      E@      I@      M@     �L@      G@      J@     �G@     �F@      H@     �H@     �F@     �G@      9@     �E@      =@      <@      B@      >@      A@      8@      (@      6@      6@      3@      4@      "@      @      @      (@      &@      "@       @      @       @      &@      @      @       @      @      @      @      @      @      �?      @      @      @      �?      �?       @              �?              �?      @      �?       @      �?              �?               @              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?              �?      �?      �?              �?       @       @      @      @              @      �?      @       @      @      @      @      @       @       @      @      (@      $@      *@      &@      (@      ,@      8@      *@      5@      3@      <@      <@      @@     �E@      C@      ;@      I@     �C@      G@     @P@      P@     �Y@      V@     �X@      `@     @Z@     �`@     �X@     �`@     @]@     �Z@      Y@     @U@     �U@      W@     @T@     �N@     �N@     �N@      K@     �D@     �A@      ;@     �A@      9@      4@      :@      1@      .@      @      "@       @      @      @      @       @      @       @       @      �?      �?      �?      @      �?              @               @               @      �?      �?              �?              �?        V(>>b2      ��K0	�}�^���A*�d

mean squared error��,=

	r-squaredx�=
�L
states*�L	      �   ���@   �$[RA! �Ү@��)HO^_Cu A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     �@      �@     4�@     |�@     8�@     0�@     8�@     ��@     ��@     ��@     ��@     }�@    ��@    @N�@    @!�@    �+�@     ��@     ��@    �;�@    @��@     ��@    �j�@    ���@     ��@    ���@     :�@    ���@    `B�@    ���@     ��@    ���@     ��@    `�@     ��@    `b�@    ���@    `)�@    @r�@    @��@    ���@    `��@    �&�@    ��@    �9�@    `��@    ���@     ��@    `S�@    `�@    `��@    ���@    �;�@    ���@    ���@    @��@    �j�@    ���@    ���@    �1�@    ���@     ��@    ���@     1�@    @��@     C�@    @]�@    �n�@    �r�@     ��@     ��@    @�@    @@�@     &�@    ���@    �!�@    �W�@     ��@     ��@    �H�@     |�@    ���@     ��@     ��@    ���@    ���@     L�@     ξ@     _�@     �@     ѹ@     M�@     X�@     ��@     �@     ��@     Ҳ@     ű@     m�@     ��@     v�@     ^�@     <�@     ��@     n�@     \�@     ��@     Χ@     �@     �@     0�@     p�@     >�@     ��@     ��@     v�@     ��@     ��@     ��@     p�@     ��@     ��@     �@     \�@     ��@     ��@     P�@     D�@     ��@     �@     ��@     ,�@     t�@     h�@     t�@     �@     Ȕ@     `�@     ̔@     <�@     (�@     ,�@     t�@     0�@     �@     ��@     ��@     H�@     4�@     L�@     $�@     h�@     h�@     Џ@     �@     X�@     ��@     ��@     ��@     Ќ@     ��@     �@     x�@     �@      �@     ��@     �@     ��@     ��@     ��@     �@     ��@     ��@      �@     ��@     p�@     H�@     h�@     �@      �@     H�@     ��@     �@     ��@     ��@     @�@     H�@     0�@     ȁ@     �@     8�@     ��@     @�@     x�@     ��@     ؀@     8�@     x�@     �}@     @     �}@     @|@     �~@     `@     �{@     p{@     z@     �y@     �{@     �y@     �y@     �{@     �z@     �{@     �z@     �y@     �u@     �x@     0z@     `z@     �x@     v@     �v@      u@     �v@     t@     u@     �t@     @s@     `s@     �s@     �t@      t@     �t@     pr@     0r@     �t@     �t@     �p@     �q@     �q@      q@     �o@     �p@     @n@     �q@      s@     `o@     q@     �m@     �p@     �p@     �o@     �p@     0q@     �p@     �g@     `i@     �j@     `k@      k@     `m@      i@     �k@     @h@     �h@     `g@     �k@     @g@     `i@      k@     �j@     �k@     �i@      j@     �e@      j@      g@     @f@     �f@     �d@     �c@     �b@     �e@     �c@     �b@      e@     �e@     �b@     �c@     `c@      e@     @a@     @c@     @`@     �_@      b@     ``@      d@     �a@     �b@     �d@     ��@     ��@     @c@      b@     @c@     @b@     `a@     �a@     �d@     �b@     @d@     `d@     �d@     �b@     �f@     @d@      f@      d@      f@     �h@     �f@      h@     �e@     �d@      i@     `f@     �f@     �g@     �f@     �f@     �f@      i@     �h@     �h@     �l@     �i@     �k@     @o@     �n@     �n@     `m@     �j@     �n@     `l@     �o@     �j@     �l@     �l@     @l@     �n@      p@      p@     `p@     �m@     �q@     �p@     �p@     �q@     �p@      q@     �q@     �q@      r@     �s@     �t@     pu@     `s@     �t@     �v@     �t@     t@     u@     �u@      u@     �u@     `v@      v@      x@     @z@     �{@     �w@     �y@      {@     �y@      y@     �{@     Px@     @{@     `z@     �{@     }@     �~@     �{@     �{@     �|@     `~@     �|@     `�@     ~@     P~@     �}@     0�@      @     ��@     p~@     ��@     ��@     �@     �@      �@     ��@     h�@     Ȃ@     p�@     ��@     X�@     �@     ��@     x�@     `�@     ؄@     �@     ��@     0�@     Ȇ@      �@     ��@     ��@     P�@     p�@     ��@     ��@     ��@     ��@     ��@      �@     ��@     ��@     �@     @�@     ؍@     (�@     �@     ��@     ��@     (�@     ��@     D�@     ��@     x�@     �@     ԑ@     D�@     h�@     l�@      �@     �@     ��@     ��@     ��@     ��@     �@     ��@     �@     �@     �@     \�@     |�@     ̘@     4�@     ��@     |�@      �@     l�@     �@     `�@     ԛ@     ��@     P�@     H�@     X�@     �@     ܟ@     ��@     ��@     ��@     *�@     ��@     *�@     h�@     n�@     �@     L�@     �@     �@     �@     d�@     
�@     ܩ@     \�@     ��@     ~�@     ��@     ��@     ��@     �@     ��@     w�@     c�@     K�@     r�@     u�@     }�@     ��@     ��@     ~�@     �@     ��@     �@    �o�@     ��@    �	�@     ��@     ��@    ���@    ���@     O�@     ��@    ���@    @��@    @��@     ��@    @U�@    ���@     ��@    ���@    ���@    ��@    �=�@    @��@    �|�@     �@    �'�@     �@    @�@    @��@    ��@    @<�@    ���@    �:�@    ��@    ���@     �@    �n�@     ��@    ���@    ��@    ��@     ��@    �k�@    ���@     2�@     r�@    `��@    ���@    ���@    ���@    ���@    @��@     ��@    ���@     ��@    `��@    ���@    @��@    `��@    �C�@    ���@    �B�@    ��@    ���@    ���@    ��@     ��@    ���@    ���@    @<�@    ���@    �y�@    �P�@    �w�@    ���@    ���@     �@     ��@     ^�@     ��@     ��@     X�@     h�@     ��@     p�@     \�@     `�@        
�
predictions*�	   �w���    4i@     ί@!  @E<3�)k��� &+@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��6�]���1��a˲�a�Ϭ(���(��澙ѩ�-�>���%�>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?�5�i}1?�T7��?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?����?_&A�o��?��7��?�^��h��?uo�p�?2g�G�A�?������?cI���?�P�1���?�6v��@h�5�@�������:�              @      @      @      4@      8@      ;@      F@     �P@     �T@      I@      V@     �Q@     @S@     �X@     �X@      W@     �V@     �\@     @X@     @W@      S@     �U@      U@     �T@     @X@     @Q@     @P@      O@     �K@     �G@      B@     �M@      E@      D@      D@      B@      A@      8@      @@      5@      =@      3@      .@      0@      0@      "@      .@      *@      "@      @      @      @      "@      "@      @      @       @       @       @      @      @              �?      @      @      �?      �?              �?              �?       @      �?              �?      �?               @              �?              �?              �?               @              �?              �?              �?               @      �?      �?      �?      �?      �?       @      @      @       @      @      �?      �?      @       @      "@      �?      @      @       @      @      @      @      @      "@      "@       @      "@      (@      0@      *@      $@      2@      4@      0@      :@      >@      ;@      A@      A@      ?@     �B@     �D@     �D@      I@      I@     �C@      O@      M@      I@     �H@      G@     �G@     �F@     �C@      J@     �B@      >@      H@      ?@      <@      >@      @@      5@      6@      "@      *@      .@      $@      ,@      &@       @      @      @      @      @      @      @       @      �?              �?              �?              �?      �?              �?              �?        ��B3      �%��	\uV_���A*�f

mean squared error��&=

	r-squaredT�>
�L
states*�L	      �   ���@   �$[RA!T�LA���)(a�&� A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             d�@     �@     L�@     h�@     r�@     Ҫ@     ��@     ��@     �@     �@     M�@     ��@    @��@    @��@    ��@    ��@     ��@    �,�@    ��@    @��@    �+�@    �m�@    @��@    ��@    ��@    ���@    ���@    `��@    ���@     ��@     ��@    ��@    `��@     ��@    ��@    �7�@    �u�@    ��@    �!�@    �8�@     ��@    ���@    @��@     �@    �*�@    �B�@     I�@    ���@     ��@    ���@    �@�@     ��@    �j�@    ���@    `f�@    @�@    �{�@    @��@    @�@     ��@     ��@    �c�@     ��@    ���@    ��@    @��@     �@    ��@    @��@     �@    ���@    �Y�@     z�@     /�@    ���@    ��@    ���@     ��@    �#�@     ��@    ���@    ���@    ���@     ��@     ��@    ���@     F�@     ��@     �@     ��@     ��@     �@     ϶@     ��@     D�@     г@     ��@     ��@     �@     ��@     ��@     ��@     �@     l�@     �@     ^�@     `�@     2�@     ��@     �@     �@     X�@     d�@     V�@     "�@     \�@     >�@     ��@     �@     4�@      �@     l�@     �@     t�@     �@     ��@     ��@     ,�@     @�@     ��@     0�@     ��@     ��@     ��@     \�@     ��@     ��@     �@     X�@     Г@     ��@     ��@     ��@     Ԕ@     ��@     ��@     D�@     ��@     ��@     ܑ@     ��@     T�@     ��@      �@     ��@     x�@     ��@      �@     �@     ��@     ��@     ��@     ��@     H�@     Ȋ@     Љ@     ��@     ��@     X�@     ��@     @�@     Љ@     ��@     ��@     ��@     �@     P�@     �@     ��@     ��@     ��@     ��@     p�@     ��@     ��@     �@     0�@     ��@      �@     X�@      �@     �@     ��@     Ё@     ��@     �@     �@     ��@     Ȁ@     �@      �@     0~@     �~@     @~@     `@     P|@     �}@     �}@     @{@     pz@     �|@     �}@     p~@     p}@      {@      |@     �x@     pt@     �x@     �x@     �u@     `u@     �v@     Pw@     Pu@     �w@     Pt@     Pw@     y@     �w@      v@     pu@     �r@     Pt@     �r@     v@     �r@     `r@     �v@     �s@      s@     `r@     �q@     �m@     0p@     �q@     �p@     p@     �o@     pr@     �p@     `o@     �n@      o@      q@     pr@     �l@      m@     �k@     �o@     �i@     �m@     @p@      h@     �j@      l@     �j@      m@     �i@     �j@     �j@     �l@      m@     �g@     �i@     @g@     �i@      h@     @h@     �d@     �f@      d@     �e@      g@     @e@     �g@     `l@     `c@      c@     �f@      e@     �e@      _@     @b@     @b@     �_@     �^@     �`@     ``@      `@     �_@     �b@    ���@     5�@     �`@      c@     �e@     �d@     `b@     �d@     `c@     �e@     `d@     �d@      e@     �d@     @g@     �f@     `g@     �f@     �e@     �f@      g@     @j@     �e@      h@     �h@     �j@     �j@     �g@      i@     �m@     �l@      p@     �n@     �m@     �l@      n@     �p@     �p@     �p@     �m@     �n@     �o@     @n@      n@     �p@      n@     `o@     �l@     0p@     �o@     Pq@     �q@     �n@     pq@     �r@     0s@     �t@     �u@     Ps@     Pr@     Ps@      q@     �s@      u@     �s@     ps@     �t@     �t@     `x@     �u@     w@     �x@     0y@      v@      w@      y@     �z@     �|@     �y@     �y@     �x@     @x@     @y@      {@     �z@     �z@     �|@     �z@     0@     �~@     ��@     �~@     �~@     �|@     p�@     P@     p}@     H�@     ~@     �@     Ȁ@     ��@     (�@     ��@     ��@     ��@     ��@     H�@     ��@      �@     ��@     ��@     ��@     ؃@     8�@     �@     ��@     ��@     ��@     ��@     h�@     Ѕ@     8�@      �@     `�@     ��@     ��@     ��@     X�@     ��@     ��@     8�@     ��@      �@     p�@     H�@     0�@     ��@     �@     �@     P�@     l�@     �@     Ԑ@     |�@     ,�@     ��@     Đ@     �@     ��@     @�@     @�@     @�@     ��@     ��@     L�@     ܔ@     �@     ��@     Ĕ@     ��@     ��@     ��@     ��@     ؖ@     @�@     |�@     ��@     ��@     ��@     ��@     ��@     0�@     ��@     ,�@     |�@     ��@     ��@     H�@     ��@     ��@     ��@     0�@     ��@     ڠ@     l�@     ��@     ��@     Ң@     �@     ΢@     X�@     £@     �@     ��@     b�@     ��@     ��@     J�@     ʪ@     �@     d�@     �@     ��@     ~�@     �@     ��@     p�@     �@     v�@     p�@     ��@     w�@     '�@     )�@     r�@     Q�@     ��@    ��@     ��@     ��@     ��@    ���@     ��@     Z�@     ��@     ��@    ��@    �J�@     ��@     �@    ���@    ���@     h�@     �@    �-�@    @�@    ��@    ���@    ���@     ��@     2�@     (�@    ���@    �R�@    ��@    @j�@     &�@    ���@    �R�@    `��@     2�@    `��@    ���@    @d�@     ��@    ���@     ��@     ��@     F�@    ���@    ���@    ���@    �)�@     ��@    �{�@    `:�@    @]�@     D�@    `^�@     ��@    �|�@     ��@     ��@    @��@     p�@    `=�@    ��@     o�@     ��@    @s�@    @�@     L�@    �p�@    @`�@    ���@     ��@    ���@    ���@     	�@     ��@    `�@    �I�@     �@     6�@     ,�@     �@     X�@      �@     �@      �@     ��@     X�@     l�@        
�
predictions*�	   `�u��    z�@     ί@!  d�2�0@)<�XS�s9@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x�������6�]�����[���FF�G �8K�ߝ�a�Ϭ(���(��澢f�����uE���⾮��%�E��a�Wܾ�iD*L�پ1��a˲?6�]��?f�ʜ�7
?>h�'�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?\l�9�?+Se*8�?+�;$�?cI���?ܔ�.�u�?��tM@u�rʭ�@�DK��@�������:�              �?               @      @       @      (@      1@      <@      >@     �F@      L@     �N@     @R@      V@     �U@     @S@     @Q@     @U@      S@     �S@     @V@     �V@     �R@      P@     �Q@     �H@     �L@     �H@     �O@     �A@      G@     �@@      ?@      D@      @@      B@      5@      6@      3@      ,@      7@      .@      1@      2@      (@      "@      1@      @      $@      @      @       @      @      @      @      @      @      �?      @       @      @       @      @      @       @      @      @      @               @               @      �?              �?      �?              �?      �?       @      �?              @              �?              �?              �?              �?              �?              �?              �?              �?              �?      �?               @       @      @      �?      �?      �?      �?      @       @              �?      �?      �?      @      �?      @      @      @      @      @      @      @      *@      (@      @      $@      @      *@      2@      &@      (@      1@      ;@      ,@      3@      ,@      2@      2@      >@      >@      :@      8@     �E@      A@      B@      C@      J@     �I@      N@      L@     �O@     @P@     @U@      N@      R@     @P@     �S@     �Q@      N@     �Q@     �J@      R@     �H@     �H@      >@      J@     �B@      E@      <@      5@      5@      5@      &@      (@      *@      "@      (@      $@      @      @      �?       @      @      �?      �?               @              �?              �?              �?              �?        �W�r2      �$A	\X�_���A*�d

mean squared errorz�!=

	r-squared\4>
�L
states*�L	      �   ���@   �$[RA!/Ȃ���)�O�t!A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     ��@     ��@     ڢ@     X�@     ��@     �@     ��@     I�@     ޹@     >�@    `��@    ���@    �b�@     �@     '�@    ���@    ���@    ���@    @�@     q�@    �.�@    �X�@    ��@    �T�@    @/�@    ���@    ���@    @��@     ��@    ��@    ��@    ��@     ��@     v�@    ���@    `L�@     ��@    ���@     �@    �o�@    �y�@     t�@     {�@    ��@    �I�@    ��@    ���@     �@     ��@    �`�@    @'�@    ���@    ��@    @��@    �V�@    ��@    ��@     �@    ��@    ���@    @r�@    @]�@    �G�@    �U�@    �z�@    �X�@     m�@    �W�@    @��@    ���@     ��@     ��@     ��@    ���@    ���@    ���@    �#�@    �e�@     ��@    ���@    ���@    � �@     ,�@     c�@     �@     �@     ��@     ��@     @�@     n�@     ��@     ��@     ��@     �@     �@     !�@     ��@     :�@     ��@     "�@     ¬@     .�@     ��@     @�@     �@     ^�@     h�@     J�@     d�@     l�@     \�@     4�@     ��@     *�@     ��@     F�@     6�@     6�@     �@     ��@     �@     T�@     p�@     �@     ��@     ��@     ��@     d�@      �@     t�@     ��@     h�@     ܖ@     ��@     �@     ԕ@     D�@     ��@     <�@     �@     `�@     ��@     0�@     ��@     �@     D�@     p�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     @�@     l�@     ��@     $�@     <�@     ��@     ؎@     ��@     h�@     ؎@     @�@     Ѝ@     ��@     ��@     ��@     8�@      �@     ��@      �@     ��@     ��@     ��@     ��@     �@     ��@     ��@     �@     `�@     ��@     H�@     ��@     X�@     h�@     ��@     Ђ@     �@     h�@      �@     P�@     ��@     ��@     (�@     ��@     (�@     ��@     P@     8�@     ��@      @     �}@      }@      @     }@     �}@      |@     @}@     �{@      |@     {@     �x@     z@     �x@     pz@     �y@     `w@     �x@     �u@      w@     �u@     `u@     0x@     �w@     Ps@      v@     �u@     �t@      t@     �u@     @w@     �t@      t@     @u@     Pu@      s@     �t@     �t@     �r@     �q@     `q@     0r@     Pr@     `r@     0p@     �n@     �p@     0q@     �q@     `r@      r@     �p@     `m@     @j@     �m@     �l@     �i@      l@     @n@     �k@     �k@      m@     �m@     @l@     @o@     �p@     �n@     `i@     `m@     @j@     @i@     �h@      h@      j@     �e@     �f@      h@     �f@      m@     `g@     `e@      c@     �e@      b@     �b@     �b@     �b@     �d@      e@     @a@     �d@     @d@      c@      a@     �a@     �a@     �b@     @c@     8�@    ��@     @^@     �b@      a@      d@     �d@     �c@     �c@      g@      e@     �b@      d@     �e@     `f@     �f@     @f@     �g@     �f@     `g@     �i@     `h@     `e@     @i@     �i@      k@      n@     �k@     �k@     �m@     �m@     �o@     �i@     �k@     `n@     @m@     �n@     �l@     p@     �l@      q@      n@     �o@     �n@     �p@      o@     �l@     0q@     �q@     @q@     �p@     0q@     �q@     �p@     �p@     0r@     `t@     0s@     Ps@     pu@     0r@     @u@     �u@     u@     Pu@     �v@     �s@     �u@     Pv@     �w@     �u@      v@     �v@     �y@      v@      x@     �{@     �{@     �y@     0z@     `{@      {@     Pz@      }@     {@      {@     �|@     �}@     P}@     p}@     �}@     �}@     ��@     P@      �@     �@     @�@     �@     0�@     �~@      �@     �@     �@      �@     X�@     @�@     �@     ��@     ��@     ��@     @�@      �@     �@     p�@     P�@     8�@     ��@     ��@     (�@     �@     h�@     ��@     ��@     ��@     @�@     �@     p�@     `�@     h�@     ؉@      �@     ��@     h�@     ��@     Ќ@     Ȋ@     `�@     h�@     P�@     8�@     @�@     Ԑ@     ��@     �@     �@     ��@     X�@      �@     �@     T�@     �@     ��@     t�@     4�@     �@     �@     ��@     ��@     �@     ��@     (�@     ��@     ��@     ؕ@     \�@      �@     d�@     �@     |�@     ��@     ��@     ��@     L�@     ��@     X�@     ��@     ԛ@     ��@     ��@     �@     ��@     L�@     `�@     L�@     H�@     �@     :�@     z�@     ��@     x�@     ��@     0�@     T�@     �@     ʥ@     *�@     ��@     ��@     ��@     l�@     &�@     �@     ��@     ��@     x�@     ��@     ��@     ��@     ճ@     F�@     ��@     ��@     q�@     �@     �@     ��@     ��@     ��@     ]�@    ���@    ���@     ��@    ���@    �9�@    ���@     ��@    ���@     l�@     �@    ���@    �=�@    ���@    ���@    �U�@    �B�@    ��@    ���@    @��@    @��@    ���@    @b�@    @I�@    �{�@    @m�@    ���@     ^�@    �R�@    �A�@     �@    `��@    ��@    `v�@    �)�@    `�@    ��@     8�@     ��@    ���@    ���@    �h�@    ���@     ��@     +�@    ���@     C�@    ���@    ���@    @��@    `��@     c�@    @��@    ���@    ���@    ���@    ��@    ���@     ��@    �[�@    ���@    ���@    ���@     {�@    @$�@    @��@    �_�@    ���@    ���@    @&�@    @��@    @*�@    ���@    ���@     ��@     `�@     ث@     (�@     .�@     n�@     $�@     ��@     H�@     X�@     ��@     ��@        
�
predictions*�	    ,rſ   `S@     ί@!  8&�vK@)��!� uC@2��QK|:ǿyD$�ſ��(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.���vV�R9��T7����5�i}1���d�r�x?�x���h���`�8K�ߝ���Zr[v�>O�ʗ��>x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?cI���?�P�1���?3?��|�?{2�.��@!��v�@�������:�              �?              �?       @      �?      @      6@      7@      :@     �F@     �M@      Q@     �S@      U@     �S@     �U@      Q@      U@     �V@     �T@     �U@     �R@     �P@     �P@     �P@      P@     �L@     �G@     �J@      K@     �I@      E@      ?@     �@@      8@      :@      :@      7@      5@      6@      6@      0@      1@      $@      0@      ,@      "@      @      "@      @      @      @      @       @      @      @      @               @      @      �?       @      �?              �?              �?      �?               @              �?              �?              �?      �?              �?              �?              �?              �?              �?              �?               @       @               @       @              �?      @       @              @      @      @       @      @       @      @      @      @      "@      @      @      ,@      @       @      "@      @      ,@       @      .@      (@      0@      2@      ;@      :@      9@      C@      :@     �@@      ;@      ?@     �C@     �C@     �B@      E@      I@     �H@      N@      K@     @P@      I@     �R@     �S@     @P@     �P@     �M@      J@     �L@     �H@      L@      K@     �F@      I@      C@     �B@      G@      D@      ?@     �B@      B@      2@      1@      2@      4@      6@      3@      $@      @      @      @      @      @       @      �?      �?      @      �?       @               @      �?              �?              �?        �?��2      ����	uZJ`���A*�e

mean squared error�$=

	r-squared�<'>
�L
states*�L	      �   @��@   �$[RA!M�_��7��)-+�dc"A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     �@     �@     ��@     f�@     �@     ��@      �@     ��@     '�@     |�@    ��@    @��@    �o�@    ���@     �@    @�@    ��@    @��@     ��@    @T�@    ��@    @m�@    �s�@     ��@    �n�@    ���@    ���@    ���@    ���@     ��@    @5�@    @D�@    `Y�@    ���@     #�@     X�@     ��@    ���@    ���@    ��@    `��@     }�@     ��@    �]�@    `�@    ��@    ���@    `��@    �t�@     4�@    @��@    �:�@    ���@    `8�@     ��@    ���@    �k�@    �w�@    �-�@     ��@     /�@    �k�@    �.�@     L�@    �Q�@    �Y�@    @T�@     ��@    @��@    ���@     X�@    �k�@     �@     ��@    �l�@     @�@     :�@     �@     y�@    ���@     ��@    �6�@    ��@     �@    ���@     ��@     *�@     ؽ@     s�@     }�@     �@     r�@     �@     ��@     S�@     |�@     O�@     �@     �@     ��@     %�@     �@     P�@     L�@     ��@     ��@     
�@     8�@     ��@     �@     &�@     �@     >�@     ��@     ��@     6�@     ��@     �@     |�@     N�@     ��@     |�@     ��@     ԟ@     �@     ��@     P�@     `�@     ��@     x�@     ,�@     �@     0�@     ��@     ��@     ��@     ��@     D�@     4�@     ��@     ��@     d�@     ��@     $�@     |�@     �@     ��@     ,�@     ��@     ��@     ��@     P�@     ,�@     ��@     |�@     0�@     8�@     ��@     Ѝ@      �@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     h�@     h�@     x�@     ��@     ��@     ��@     �@     H�@     p�@     ��@     �@     ��@     P�@     H�@     ��@     ��@     ��@     Є@     ؃@     ��@     x�@     ��@     ��@     `�@     ��@     ��@     h�@     8�@     �@     p�@     h�@     0�@     �@     ��@     �}@     �}@     �}@     P|@     �z@     Px@     �|@     �{@     `|@     p|@     �|@     �{@     Pz@      y@     �x@     �x@     �y@     �v@     �w@     @w@     pw@      w@      u@     pu@     �t@     �q@      s@     �t@     �v@     �q@     �t@     �t@     0s@     �r@     @t@     �u@      v@     `r@     @r@     �q@     r@     Ps@     �q@      r@     r@     �r@     @t@     p@     0p@     �n@     Pp@     �l@      o@     �o@     �i@     �m@      l@      k@     �i@      j@     `k@     �m@      k@     `m@     �p@      m@      h@     �e@     `h@     �j@     @i@     �i@     �e@      g@      f@     �g@      f@     `g@     �h@     �g@     �i@      d@     `c@     �a@      b@     @e@      b@     @d@     �`@     `b@      c@      `@     ``@      c@     �`@     �d@     �b@     �_@     �_@    ���@     O�@     @b@      f@     �c@     �c@     �g@     @a@     @b@      d@     `e@      f@     @f@     �d@      e@     �h@     �d@     `f@     `g@     `h@      i@     �g@     �k@     @j@      i@     �f@     �i@     �h@     `h@      i@      i@     @h@     �j@      j@     �l@     pp@     �m@     �p@     Pp@     �o@     �n@     �n@     p@     �p@     `m@     �o@      r@      s@     pq@     0p@     p@     0p@     Pq@     �p@     �p@      q@     Pp@     �r@     �p@     �q@     �q@     @s@     �u@     �q@     pw@     �t@     �s@      w@     pu@     pw@     �t@     Pv@     �t@     �w@     @x@     �w@     �y@      x@     �x@      y@     �y@     �z@     `{@     �y@     �z@     P|@     �|@     �x@     p}@      }@     �{@     �@     ��@     p�@     �@      @     �@      �@     �@     8�@     H�@     @�@     ��@     x�@     ��@     ��@     (�@     ��@     P�@     (�@     ��@     ��@     ��@      �@     P�@     x�@     �@     ��@     x�@     h�@     ��@     h�@     H�@     ��@      �@     ��@     ؇@     ��@     ؇@     �@     @�@     ��@     Њ@     ��@     P�@     �@     �@     8�@     H�@     ��@     L�@     ��@     ��@     \�@     X�@      �@      �@     <�@     T�@     А@     Đ@     <�@     p�@     L�@     �@     ��@     P�@      �@      �@     h�@     Г@     Д@     ��@     ��@     h�@     ��@     �@     ܗ@     ��@     �@     d�@     ��@     |�@     ��@     0�@     ̚@     �@     ��@     Ԝ@     X�@     ��@     �@     ��@     Z�@     X�@     T�@     ��@     ȡ@     �@     L�@     �@     �@     8�@     Υ@     ��@     �@     R�@     ֩@     .�@     ��@     F�@     �@     R�@     �@     G�@     }�@     B�@     �@     Z�@     d�@     �@     �@     ��@     �@     C�@     �@     �@     ��@     ��@     I�@     y�@    ���@     ��@    ���@    ���@    �g�@     *�@     ��@    ��@     ��@     V�@    �R�@    ���@    @��@    �^�@    ��@    ���@    ���@    @r�@    �H�@    ���@    @��@     ��@    ���@     ��@    @��@     b�@    @�@     _�@    �7�@     ��@    ���@     s�@     ��@    `-�@    `��@      �@    �^�@    �p�@    �=�@     A�@    ���@     E�@     ��@    `q�@    ���@    @��@    ���@    �g�@    @]�@    ���@    ���@     ��@    ���@    @��@    ���@    ��@    �x�@     '�@    ���@     ��@    @��@    ���@    ���@    @z�@      �@     �@    �b�@    ���@     ��@    ���@    `F�@     ��@     \�@     ��@     ^�@     �@     F�@     �@     ��@     @�@     ��@     h�@     h�@        
�
predictions*�	   � ���    �@     ί@!  PMdz7�)�8�,C@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���vV�R9��T7���f�ʜ�7
������pz�w�7��})�l a���>M|K�>�_�T�l�>})�l a�>pz�w�7�>6�]��?����?f�ʜ�7
?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?h�5�@�Š)U	@�DK��@{2�.��@�������:�              �?      @      @      @      &@      <@     �B@      G@     �L@     �T@      U@     �_@     �Z@      ^@     �a@     �b@     �a@      c@     �`@      `@     �\@     �_@     @\@     @V@      Y@      T@      S@     �O@     �M@     �M@     �K@      I@     �A@      B@      @@     �@@      6@      8@      9@      5@      3@      .@       @      .@      @      @      .@      @      @      @      @      @      �?      @      @      @      @      @      �?      @      @      @      @      @      �?      �?      �?      �?      �?      �?              �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              @      @       @               @      �?      �?      @      @      �?      @      @       @       @      �?      @      @      @      @      @      @      @      @      @      @      &@      @      $@      &@      &@      (@      (@      *@      *@      1@      &@      0@      <@      9@      <@      ;@      :@      5@      :@      5@      =@      ?@      =@      9@      <@     �B@      ;@      @@     �C@      <@      @@      C@      :@      =@      >@      8@      7@      :@      3@      *@      .@      0@      $@      *@      $@      @      $@      @      @      @      @      @      @      @       @              �?      @       @      @              �?              �?               @              �?              �?              �?        n�L�2      ��!s	��`���A*�c

mean squared errorI%=

	r-squaredT!>
�L
states*�L	      �   ���@   �$[RA!��Dhz��)ˑ�M2#A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     Ԓ@     ��@     Τ@     ԯ@     �@     �@     �@     ��@     �@     )�@    �Z�@    ���@    � �@    `��@     ��@    ���@    @s�@     U�@    �2�@    �E�@    @�@     <�@    ��@    @d�@    �v�@    @`�@    �-�@    �n�@    `a�@    �e�@     ��@    ���@     ��@    � �@    �q�@    ���@     c�@    ���@     ^�@     ��@     ��@    ���@    ���@    �D�@    @�@    `��@     &�@     ��@    `��@     �@    @��@    ���@    `4�@     ��@     �@    �b�@     x�@     ��@    @��@     ��@    ���@    @Y�@     $�@     K�@    �4�@    �V�@     ��@    @�@     �@    �B�@    ���@     ��@    �_�@     ��@    ��@    �0�@    ��@     �@    ���@     ��@    ���@     ��@     1�@    ���@    �B�@     J�@     d�@     ¿@     �@     ��@     �@     R�@     /�@     ��@     ��@     ��@     I�@     �@     �@     ��@     '�@     1�@     �@     ��@     ��@     ��@     ��@     T�@     ��@     �@     "�@     .�@     �@     .�@     N�@     >�@     <�@     h�@     `�@     �@     ��@     �@     ��@     ��@      �@      �@     �@      �@     ��@     ��@     l�@     h�@     ��@      �@     ̖@      �@     �@     ,�@     ��@     4�@     �@     ��@     �@     �@     �@     ��@      �@     �@     ��@     H�@      �@     ��@     d�@     �@     `�@     @�@     �@     ��@     @�@     8�@     Ȏ@     8�@     ��@     ȍ@     x�@     H�@     ��@     ��@     8�@     P�@     ��@     ��@     8�@     ��@     ��@     ��@     ��@     8�@     ��@     ��@     ��@     x�@     x�@     0�@     ��@     p�@     ؂@     @�@     8�@     `�@     8�@     ��@     �@      �@     (�@     �@     Ѐ@     `�@     ��@     �|@     @@     �|@      }@     P}@     �~@     �|@      |@     �x@     �z@     Py@     �y@     �y@     Py@      z@      w@     pw@     �w@     0w@      t@     0z@     �v@     �w@     �u@     `w@     u@      t@     `t@     �r@     �u@     �u@     �s@     �s@     0v@     0t@     �r@     �r@     ps@     t@     �s@     `t@     �r@     �r@     �q@     Pr@     ps@     �q@     �o@     �p@     �p@     �p@     �q@     �p@     �o@     �p@     �l@     �n@     `j@      i@     `l@     �k@     �l@     �j@      k@     �k@     �k@     �h@      g@      h@     `i@     �h@     �f@      e@      d@     �e@      d@      g@     @h@     @g@      f@      g@      h@     `f@     �g@     @c@     `b@     �b@      b@      a@      `@     @b@     �c@     �`@      a@     �a@     �a@     �_@     @a@     �a@     �a@     �a@    �S�@     ��@     @e@     @_@     @c@     �e@     �`@      e@     @b@     @d@     �a@     �f@      f@     �e@     �b@     �d@      f@      f@      j@     �i@     @j@     `h@     �n@     `h@     `g@      h@     �j@      i@     @h@     `j@     �h@      l@     �i@     @l@     �l@     �m@     �m@     �j@     �n@     �p@      n@     @n@     Pp@     `p@     `o@     �o@     @n@     �m@     0p@     @o@     �p@      r@      n@     Pp@      r@     �p@      r@     �p@     �r@     �r@     �s@     �r@      t@     �r@      t@     0t@     `u@     �t@     �t@     �v@      t@     `u@      u@     �w@     �w@     w@     �w@     �z@     px@     z@     `x@     �y@     �x@      y@     `{@     p}@     �{@     p}@     {@     �{@     P~@      @     �z@      }@     �}@     �~@     �~@     �@     @�@     ��@     `�@     �@     0�@     ��@     ��@     ��@     ��@      �@     P�@     x�@     ��@     ��@     ��@     �@     8�@     p�@     h�@      �@     h�@     0�@     ��@     ��@     `�@     (�@     Ї@     ��@     ��@     x�@      �@     H�@     Ј@     ��@     h�@     �@     (�@     ��@     P�@     ��@     (�@     x�@     ��@     Ȏ@     �@     ��@     4�@     h�@     �@     �@     �@     ��@     ��@     ��@     ��@     ��@     @�@     ��@     ��@     �@      �@     x�@     �@     ,�@     4�@     �@     �@     ��@     ��@     4�@     d�@     �@     ��@     t�@     0�@     d�@     �@     @�@     ��@     �@     ԝ@     Ԟ@     �@      �@     �@     f�@     �@     ��@     ޢ@     ̣@     ��@     �@     6�@     �@     ��@     *�@     ��@     �@     ��@     ��@     �@     ��@     �@     ��@     ��@     C�@     ��@     ´@     ��@     O�@     ��@     ��@     +�@     ��@     r�@     �@    �+�@     ��@    ���@    ���@    ���@    �w�@     ��@    ���@     c�@     ��@     ��@    �C�@    ���@    ���@    ���@    �K�@    @c�@     5�@    ��@     ��@    ��@     ��@    �s�@    �}�@     �@    ���@    �u�@    �X�@    ���@    ���@    �v�@    �V�@    �]�@     ?�@     ��@    @��@     \�@    �#�@    �k�@     ��@    ��@    �I�@     t�@    ��@    �r�@    ��@     ��@    @4�@    ���@    @@�@    ���@    @0�@    @&�@    ���@     X�@    @��@     ��@    �c�@     ��@    �R�@    �O�@    �S�@     &�@    ���@    ���@    �W�@     4�@    �5�@     @�@    �O�@     m�@    ���@    �|�@    ���@    �%�@    @��@    �l�@    @R�@     ~�@     "�@     >�@     ��@     ��@     v�@     У@     �@     ��@     ��@     ȓ@     �@        
�
predictions*�	   `��   ��@     ί@!  ��NFT@)��9F6@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$�ji6�9���.����ڋ��vV�R9��T7����5�i}1�1��a˲���[��>�?�s���O�ʗ�����~��¾�[�=�k��1��a˲?6�]��?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?u�rʭ�@�DK��@�������:�              �?              �?               @      @       @       @      @      (@      *@      4@      7@      :@      <@      8@     �A@      A@     �G@     �D@      ;@      >@      5@      4@      ;@      8@      >@      7@      (@      :@      7@      5@      9@      *@      *@      &@      .@      ,@      (@      (@      .@       @      &@       @      @       @      $@       @       @       @      �?      @      @      @      @      @              @      @       @       @       @      �?              �?       @      �?      �?               @      �?      �?       @      �?              �?               @              �?              �?              �?      �?              �?      �?              �?               @              �?       @              �?      �?      �?              �?      �?      $@      @      @      @       @      @       @      @       @      @      "@      .@      &@      "@      ,@      0@      1@      1@      <@      7@      2@      8@      >@     �C@      K@     �J@     �P@      P@      V@     �U@     �X@      ^@     �^@     �b@     �c@     �f@     @c@     �b@      `@     �a@      `@     @Z@      Y@     �Y@     �X@     �T@     @R@     �M@      N@      ?@     �@@      <@      ;@      *@      ,@      1@      *@      .@      @       @      @       @       @               @      @      @      @      @      �?      @       @               @              �?        ��2      {�T	i�Ia���A*�e

mean squared error��#=

	r-squared��'>
�L
states*�L	   ����   ���@   �$[RA!I��.��)��6�F$A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Э@     ��@     �@     �@     ��@     ��@     ��@     1�@     B�@     q�@     \�@    ���@    ���@    @��@     ��@    ��@    ��@    �1�@     �@     ��@     K�@    ���@    ��@    �\�@    @��@     ��@    ���@    `��@    @��@     ��@    ���@    @�@    �8�@     �@     F�@    �0�@    ���@    ���@    �B�@    ���@    ���@    ���@     ��@    `E�@    @6�@    ���@     ��@    �2�@    `��@    `��@    �^�@    @/�@     �@    @"�@    �[�@    ��@     K�@     \�@     D�@    ���@    �*�@    @��@     %�@    �4�@    @>�@    �A�@    ���@    ���@    ���@    �0�@     ��@     ��@     ;�@    �C�@    ��@     Q�@     ��@    ���@     9�@    ���@    ��@    ���@    ��@    �B�@    �?�@    �8�@    �w�@    ���@    ���@     �@     9�@     ��@     �@     ��@     ��@     �@     i�@     �@     D�@     b�@     X�@     �@     ˲@     ұ@     �@     ��@     �@     \�@     $�@     ��@     ��@     ��@     R�@     L�@     ��@     �@     x�@     ��@     ��@     X�@     �@     ��@     t�@     ¡@     4�@     ��@     �@     �@     ��@     D�@     ț@     Ě@     @�@     ��@     ԙ@     ��@     ��@     (�@     ��@     Е@     x�@     ��@     �@     ؔ@     x�@     H�@     Ē@     ��@     �@     ��@     D�@     ̐@     В@     �@     ��@     (�@     ��@      �@     ��@     @�@     ��@     @�@     x�@      �@     �@      �@     �@     0�@     P�@     Ј@     ��@     ȇ@     p�@     ��@     �@     h�@     ��@     ��@     8�@     ��@     І@     @�@     Є@     �@     h�@     H�@     P�@     h�@     Ђ@     ��@     H�@     H�@     X�@     8�@     `�@     �@     H�@     ��@     ��@     Ѐ@     ��@     p|@     �~@     ~@     0}@     �}@     `|@     @z@     �{@     �x@     z@      z@     �y@     �x@     0x@     �w@     @w@     v@     �w@      v@     Pu@     �v@     @t@     @v@     �u@     �s@     t@     pu@     �u@     pt@     Ps@      t@     `t@     �t@     �u@     �r@     ps@     �r@     �t@     ps@     �s@     �s@     `s@     �s@     �q@     �s@     `s@     �q@     �q@      q@     �n@     �q@     @m@      o@      m@     �n@     @n@      g@     `i@     `k@     �l@     �k@      k@     �h@     �h@     �f@     `g@      k@     �e@     @f@      h@     �h@      i@     `c@     �h@     @l@     �h@     �h@     �i@     @f@     �f@      h@     �b@      c@     @e@     �d@     �`@     @a@      `@     @b@     `e@      d@     �b@     @_@     @Z@     @^@     �`@     �a@     �_@     �c@     @d@      d@     ��@     $�@     `i@     @f@     �e@     @e@      e@     `c@     @e@      d@      e@     @e@     @f@      i@     �h@     �e@      h@      i@     �h@     �m@     �l@     �d@     �e@     �i@     �j@     �j@      m@     �i@     �k@      j@      k@     �i@     �l@     `m@     �q@     @k@     @o@     @l@     @n@     @p@     �n@     �m@     �p@      m@     �l@     �m@     @r@     �n@     �n@     �o@     �q@     @o@     �r@     �q@     �q@     �p@     Ps@     �q@     r@     �q@     @s@     �q@     @s@     �t@     �t@      u@     �u@     �v@     u@     �u@     `u@     �w@     v@      w@     @w@     �v@      x@     y@     �x@     @x@     �|@     �w@     px@     z@     �|@      }@     �|@     �|@      }@     �@     �}@     p}@     P}@     0}@     �~@     ��@     �@     �~@     P�@     x�@     Ѐ@     h�@      �@     Ѐ@     �~@     ��@     (�@     �@     ��@     0�@     H�@     (�@     x�@     �@     (�@     �@     ��@      �@     H�@     ��@     ��@     @�@     ؅@     X�@     h�@     ��@     P�@     8�@     ��@     h�@     `�@     ��@     P�@     ؊@     ��@     ��@     0�@     ؍@     ��@     `�@     �@     ��@     (�@     �@     ��@     ��@     Ѝ@     X�@     `�@     ��@     0�@     x�@     ��@     �@     ��@     |�@     ��@     @�@     ��@     x�@     ��@     H�@     ܕ@     ��@     ��@     Ė@     ԗ@     p�@     ��@     ��@     `�@     Ԛ@     h�@     ��@     ��@     ��@     ��@     P�@     `�@     6�@     ��@     �@     N�@     �@     x�@     �@     R�@     ��@     ¦@     F�@     Z�@     ,�@     ��@     "�@     l�@     |�@     *�@     �@     α@     I�@     ̲@     �@     9�@     �@     .�@     Y�@     _�@     ��@     �@     �@     ��@     a�@     ��@    ���@    ��@     �@     ��@    ���@    �~�@    ���@    �c�@     [�@    ��@    ��@    ���@     L�@     ?�@     ��@     ��@    ���@     l�@    @f�@     ��@     ��@     �@    ���@    �x�@    ��@    ���@    ���@    @�@    ��@    @��@    @��@    ���@    ���@    ���@    ���@    ���@     `�@    @K�@    �-�@     ��@    @��@    @��@    ���@    �)�@    �8�@    ��@    ���@     ��@    @��@     b�@    �*�@    `"�@     ��@    @��@     ��@    �|�@    @��@    @L�@    ���@     %�@    @X�@     Q�@    �}�@    ���@    �U�@    @�@    ���@    ���@     ��@    @l�@    ���@    @��@     Y�@     Z�@    ��@     ��@    ���@    @��@    �{�@     ��@     G�@     ��@     �@     ��@     ��@     ֥@     X�@     <�@     ��@     H�@     ��@        
�
predictions*�	   �G	Ŀ    �@     ί@!  ~�yEI�)�F�C6@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��O�ʗ�����Zr[v��I��P=��pz�w�7���ѩ�-߾E��a�Wܾ�ѩ�-�>���%�>�f����>��(���>8K�ߝ�>�h���`�>>�?�s��>�FF�G ?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?uo�p�?2g�G�A�?+�;$�?cI���?u�rʭ�@�DK��@�������:�              �?      �?      @      (@      *@      C@      >@     �C@     �G@     �J@      T@      P@      T@     �_@      Y@     �\@     @`@      `@     �a@     �b@     ``@      _@     �`@     @^@      `@     @\@      ]@     �X@     �V@     @R@      R@      O@     �I@      M@      B@     �H@     �B@      8@      5@      <@      5@      3@      ,@      5@      (@      1@      0@      &@       @      "@      $@      "@      @      @      @      @      @       @      @      @      @      @       @      @      �?      �?       @      �?              �?              �?      �?      @              �?      �?              �?               @              �?              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?       @               @       @      @               @      @      @      @              @      @      @       @              @      @      @      @      @      @       @      @      *@      &@       @      $@      &@      (@      ,@      1@      &@      4@      ,@      5@      1@      4@      3@      7@      4@      :@      6@      =@      ?@      *@      :@      4@      7@      1@      7@      6@      =@      9@      9@      4@      0@      6@      5@      0@      0@      $@      ,@      &@      *@      (@      "@       @      @       @      @      @      @              @      @       @      �?      @      @      �?              �?      �?       @              �?              �?              �?        H'�(�3      +|�	Lp�a���A*�g

mean squared error� =

	r-squaredX$8>
�L
states*�L	   ����    ��@   �$[RA!B�`���)d�-���#A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ̘@     \�@     l�@     \�@     �@     @�@     H�@     m�@     Ҹ@     ��@     %�@    P��@    ���@     y�@    �C�@     <�@    @��@    ��@    ���@     ��@    �T�@    @p�@    @p�@    �'�@    �#�@    `�@    @��@     ^�@     ��@    ���@    ���@    �2�@     N�@    ���@    ��@     5�@    ���@    ���@     ��@    ���@    ���@    ���@    ���@    ��@    �m�@     �@    ���@    `��@    �R�@    ���@    ���@    �B�@    �/�@     O�@    �r�@    ��@     ��@    ���@     ��@    �B�@    �e�@    ���@    ���@    ���@     ��@    ���@    @��@     �@    ���@    ���@    @��@    @��@    �l�@     '�@    ��@    ���@     �@    ���@    �?�@     ��@     ��@     ��@    ��@     ��@    �X�@    �d�@    �$�@     ��@    ��@     G�@     ��@     ��@     ��@     �@     �@     K�@     ��@     c�@     ��@     ,�@     ˲@     ��@     ��@     t�@     L�@      �@     @�@     L�@     .�@     ��@     ȩ@     ��@     �@     :�@     8�@     ؤ@     ԣ@     8�@     b�@     T�@     b�@     �@     ��@     .�@     ��@     �@     H�@     ��@     ��@     h�@     ��@     T�@     ��@     ��@     Ș@     |�@     $�@     �@     �@      �@     0�@     �@     X�@     p�@     ��@     �@     Ȓ@     ��@     X�@     В@     ��@     0�@     ��@     Ԑ@     ��@     T�@     h�@     ،@     T�@     H�@     Ў@     ،@     X�@     ��@     ��@     8�@     ��@     �@     �@     P�@     ؇@     ��@     p�@     `�@     ��@     x�@     ��@     ��@     ��@     Ȇ@     x�@     �@     x�@     X�@     ȃ@     8�@     ��@     ��@     Ђ@     ��@     �@     H�@     8�@     �@     0�@     P@      �@     8�@     �~@     �~@     `}@     �|@     `@     �~@     �}@     �|@     �|@     `}@     P}@     `{@     �{@     �x@     �x@     0x@     z@     w@     �y@     @x@     y@     0w@     �x@     `v@     �w@      u@     px@     �v@     �s@     �u@      v@     0s@      v@      v@     �s@     Pt@      s@     0t@     �r@     0t@     �u@     pt@     �s@     �r@     �q@      p@     �q@     0r@     �q@     q@     @p@     �n@     �o@     �p@     @n@      o@     �o@     @l@     `m@     �m@      f@     �k@     �j@     �i@     �j@     �k@     @l@     `n@     �h@     @j@     �h@     `l@     �j@     �j@     �h@      e@     `f@     `k@     `f@     �i@     `j@     �d@     �e@      c@     @g@     �c@     �b@     �d@     �d@     �c@     @d@     �b@     �b@     @c@     `c@     `b@     @_@     �c@     �`@      _@      d@      f@     �f@    ���@    �T�@      d@      f@     �f@     �d@     `c@      e@     �f@      i@      i@     `e@      f@      f@      d@      h@     �g@     `j@     �i@     �j@     �g@     �i@      i@     �l@     �j@     @i@     �j@     �k@     �k@     `n@     �k@     �n@      o@     pq@     @o@     �m@     �q@     `j@     �k@     `l@     �m@      q@     �o@      p@     �m@     �o@     �p@     @q@     �p@     �q@     0r@     �q@      q@     �p@     �r@     �p@     ps@     �r@     �q@     �r@     �u@     �t@      u@     �t@     `t@     �t@     �u@     �u@     pu@      w@     0v@     pu@     @v@     �y@     �w@     �w@     �w@      y@     �w@     `y@     0}@     �y@     �z@     �z@     �y@     P|@      }@     �}@     0}@     �@     @@     �@     @@     p�@      �@     8�@     H�@      @     �@     ��@     X�@     �@     P�@     (�@     h�@     ��@     h�@     ��@     Ё@      �@     ��@     H�@     Є@     ��@     p�@     ��@     ��@      �@     �@      �@     ��@     ��@     (�@      �@     ��@     @�@     ��@     ��@     �@      �@     �@      �@     ��@     ��@     P�@     ��@     ��@     �@     X�@     ؎@     ��@     ��@     (�@     0�@     X�@     4�@     ��@     �@     8�@     Ԑ@     ��@     @�@     ,�@     ��@     H�@     ��@     �@     ��@      �@     ��@     0�@     t�@     ��@     8�@     @�@     ̖@      �@      �@     ��@     �@     �@     4�@     ��@     @�@     ��@     \�@     �@     ��@      �@     ��@     d�@     ��@     ��@     ��@     ��@     0�@     �@     �@     �@     H�@     ��@     ��@     ��@     ީ@     r�@     D�@     ��@     x�@     ��@     �@     M�@     J�@     �@     �@     �@     µ@     ��@     ��@     �@     ��@     ��@     �@     )�@     �@     ��@    �J�@    ���@     )�@    ���@     �@     ��@    �t�@     ��@    ���@     ��@     \�@    �o�@     k�@    ���@     ,�@    @{�@     $�@    @��@    �U�@    �N�@    @��@    ���@     �@    @#�@    ���@     g�@    �P�@     +�@     �@    @�@    ���@     ��@    ��@    @ �@    ���@    �a�@    @��@    `�@     ;�@    ���@    ���@    `�@    `��@    `��@    ��@     t�@    `
�@    @��@    �s�@    @��@    @5�@     ��@    �<�@    @P�@    ���@     ��@    �o�@    �c�@    �`�@    @*�@     ��@    ���@    �0�@     ��@    @��@    @K�@    @��@    �^�@    @��@    ���@     ��@     &�@    �{�@    ��@    @Q�@    �L�@     Y�@     k�@     �@     ��@     ��@     �@     �@     b�@     Ĝ@     X�@      �@     �@      �@        
�
predictions*�	   ��嵿   @��@     ί@!  !��Q@)�1���@@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7�����d�r�x?�x��>h�'��f�ʜ�7
�6�]���1��a˲���[��a�Ϭ(���(��澢f����0�6�/n���u`P+d��豪}0ڰ>��n����>�����>
�/eq
�>;�"�q�>['�?��>E��a�W�>�ѩ�-�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?S�Fi��?ܔ�.�u�?�DK��@{2�.��@�������:�              �?      @      @      @      $@      (@      2@      <@     �A@      A@     �E@     �A@      I@     �J@      E@     �K@      F@     @Q@     �M@     �J@     �H@      J@     �L@      H@      G@      J@      E@      G@     �B@      A@      D@      >@      A@      1@      6@      6@      7@      8@      *@      *@      $@      "@      .@      &@       @      @      $@       @      @      &@      @      @      @      @      �?       @      @      �?      @       @      @       @       @       @       @      @      @      �?      �?              �?              �?      �?              �?      �?      �?              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?      �?       @              �?              �?               @      �?      �?      �?      �?      �?      @      @       @       @              @      �?       @      @      @      "@      @       @      @      @      @      (@      @      @      3@      6@      ,@      3@      3@      5@      &@      7@      9@      ?@      <@      E@     �A@     �F@     �F@      J@     �L@     �M@     �P@     �R@     @T@     �U@     �Y@     �Q@     @T@     �R@     @U@     @S@     @S@     �R@      U@     �P@     �P@     @Q@     �M@      O@     �P@     �I@     �D@      A@     �B@      D@     �B@      =@     �@@      8@      :@      .@      @      (@      "@      &@      @      @      @      @      @      @              @      @       @               @               @      �?      �?              �?      �?              �?              �?        <���2      �m�	8�Ib���A*�e

mean squared errorX�!=

	r-squared��3>
�L
states*�L	   ����    ��@   �$[RA!��豃��)Hd=��.$A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     ܓ@     T�@     ��@     ��@     W�@     @�@     װ@     p�@     �@     �@     l�@     ��@    ���@    `��@    ���@     P�@     �@    ��@    �}�@    ���@    �0�@    �+�@    ���@    �-�@     
�@    �'�@    @�@    ���@    ���@     ��@     �@    @=�@     M�@    @��@     ��@    `G�@    �u�@    ��@     ��@    �$�@    ���@    @��@    ���@    ���@     ��@    ���@    �G�@    @"�@    ���@     ��@    �E�@    @�@    �f�@     G�@    �,�@     ��@    @��@     �@    �1�@    ���@    ���@    @��@    ���@    �+�@    �~�@    ���@    ���@    @��@    @�@     D�@    ���@    ��@     ��@    ���@    ��@     ��@     ��@     T�@    �J�@    �i�@    ���@    �q�@     �@     �@     ��@     b�@     ��@     u�@    �)�@     ��@     :�@     1�@     ��@     ׺@     P�@     T�@     �@     �@     F�@     ��@     ��@     !�@     �@     ��@     į@     d�@     ��@     ��@     �@     b�@     ��@     �@     �@     ��@     ʦ@     8�@     �@     ,�@     ��@     b�@     j�@     t�@     ��@     �@     �@     d�@     ��@     ��@     ��@     (�@     ��@     ��@     ��@     H�@     ��@     P�@     ĕ@     ��@     ��@     �@     �@     ��@     �@     h�@     Ԓ@     h�@     $�@     đ@     ��@     ��@     $�@     ԑ@     t�@     x�@     ��@     �@     4�@     (�@     x�@     p�@     ��@     0�@     0�@     x�@     ��@     �@     X�@     ��@     H�@     �@     ؆@     @�@     �@     ��@     ��@     x�@     ��@     Ї@     (�@     �@     H�@     Ȅ@     Ѕ@     ȅ@     `�@     @�@     �@     ��@     ��@     P�@     h�@      �@     Ё@     �@      �@     ��@      �@     ��@     0�@     `@     �~@      �@     �}@     �}@     }@     @     �~@     �z@     P{@      |@     0|@     0{@     �|@     �x@      w@     �w@     �y@      z@     y@     pu@     �v@      u@     `v@     �v@     pv@     �w@     �u@     �r@     pu@     ps@      u@     pr@      s@     pr@     �s@     �r@     �r@     @t@     r@     �s@     �r@     �s@     �q@     �p@     �r@     0t@     �q@     �p@     �q@     �o@     q@     �p@      p@     �m@     �m@     �m@     `k@     �k@      m@     @k@      m@      m@     �j@     �j@     �n@      h@      m@     �h@     �h@      h@     �i@     �n@     �g@     �h@     �g@     `i@      i@     �i@     `j@     `f@     �c@     �d@     �f@     �b@     �c@     �d@      d@      e@     @b@     �c@     �b@     �_@     @b@     `a@      `@     �a@     `c@     @`@     �\@     �_@      c@     ?�@    ���@      d@     �b@     �b@     `e@     @f@     �f@      e@     �e@      h@      g@      h@     �g@      i@     �l@     �h@     `i@      h@     @h@      h@     �k@     �h@     �h@     @j@     `o@     @o@      o@     �k@     �l@     �l@     @m@     @n@     `j@      h@     �n@      m@     �n@     @m@     �q@     �n@     �p@      o@     �o@     pp@     �n@      n@     �o@     �p@     @s@     @n@     `r@     �p@     `r@     �r@     �s@     �r@     �r@     �q@      t@     �s@     �s@      t@     t@     `t@     �w@     �t@     pw@     0v@     �t@      w@      x@     `|@     �x@     �v@     �w@     �w@     Pz@     �w@      w@     `x@     0x@     �z@     �z@     �|@     �{@      z@      }@     �{@     0@     �~@     ��@     ��@     @�@     �@     ��@     x�@     8�@      �@     ��@     p�@     8�@     0�@     @�@     ��@     h�@     ��@      �@      �@     ��@     Ђ@     ��@      �@     Ѓ@     (�@     ��@     �@     �@     ��@     ��@     ��@     X�@     ؆@     (�@     0�@     p�@     Ј@     `�@     �@     �@     H�@     �@     0�@     ��@     �@      �@     @�@      �@     ��@     (�@     @�@     ��@     ��@     ��@     �@     ��@     �@     $�@      �@     А@     T�@     �@     �@     h�@     �@     D�@     �@     P�@     $�@     Ԕ@     t�@     ��@     ��@     T�@     t�@     H�@     �@     ��@     ��@     ��@     �@     @�@     (�@     ��@     ��@     �@     l�@     0�@     8�@     ��@     n�@     �@     ȡ@     ��@     ��@      �@     �@     ��@     |�@     0�@     ��@     �@     *�@     ��@     ��@     ��@     ��@     )�@     Ͱ@     �@     ��@     ]�@     6�@     P�@     ��@     ��@     @�@     I�@     ��@     ��@     ټ@     ;�@     ��@     ��@     Y�@    ���@     �@     ��@     ��@    �^�@     ��@    ���@     ��@     ��@    ���@     ��@    �M�@    ��@    �g�@     )�@    @��@    @g�@    @��@     ��@    �X�@    @��@    ���@    �;�@    � �@    ��@    ���@    �t�@     m�@    @N�@    @�@    ���@    @Q�@    �e�@    �Y�@    �'�@    ���@     ��@    @3�@    �E�@    @��@     p�@    �d�@     u�@     �@    ���@    �z�@    �`�@    ��@     ��@    �y�@     ��@    �h�@     D�@    �"�@    ��@    ���@     �@    @(�@     ��@    ��@    ���@     h�@    @,�@     ��@     ��@    @t�@    @l�@    ���@     +�@    ���@    ���@    ���@     ��@     ��@    @L�@    @��@    @��@    ���@     O�@     5�@     8�@     D�@     ̫@     ��@     ��@     \�@     ��@     ��@     �@        
�
predictions*�	   �5���    :�@     ί@!  ȻVK?@)��`6�?@2���(!�ؼ�%g�cE9���{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�x?�x��>h�'��I��P=��pz�w�7���h���`�8K�ߝ뾮��%ᾙѩ�-߾a�Ϭ(�>8K�ߝ�>pz�w�7�>I��P=�>��[�?1��a˲?6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?\l�9�?+Se*8�?uo�p�?2g�G�A�?S�Fi��?ܔ�.�u�?!��v�@زv�5f@�������:�              �?              @      @      @      2@      0@      ;@      :@     �D@      K@     �N@      Q@     �P@     �R@      R@     �S@     �S@     �S@     �P@      U@     �U@     �S@     �U@     �N@     @Q@      H@      K@      K@     �B@      E@     �D@      A@     �D@      ;@      :@      A@      ;@      5@      0@      $@      ,@      2@      .@      1@      1@      (@      $@      *@      "@      @      $@      "@      "@      @       @      @      @      @              @       @      �?      �?      �?      �?       @      �?              �?      �?              �?              �?              �?               @              �?              �?              �?              �?      �?      �?               @      �?               @      �?       @      �?      @      �?      @      @      @               @      �?       @       @      @      �?      @      @      @      @      @       @      @       @      .@      *@      $@      ,@      &@      8@      2@      8@      ,@      9@      7@      6@      =@      A@      >@     �C@     �H@      >@     �F@     �J@     �G@      O@      I@      Q@      L@      R@      M@      J@      D@     �M@      K@      N@      H@      N@      F@      H@     �I@     �E@     �G@      H@      G@     �B@      D@      >@      =@      :@      3@      1@      8@      1@      2@      (@       @      *@      $@      @      @      @      �?      @       @       @      @              �?              �?               @              �?              �?        $�mv�2      ����	���b���A *�e

mean squared error-)=

	r-squared�c>
�L
states*�L	   ����    ��@   �$[RA!h��޹��)���YX$A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     ��@     �@     L�@     x�@     ��@     ��@     ��@     ߲@     k�@     ?�@     ��@    ���@    ���@    `:�@    �N�@    �6�@    ���@     ��@    ���@    ���@    ��@    ���@     f�@     ��@     ��@    ���@    �{�@     ��@     ��@    @��@    `�@    � �@     ^�@    �r�@    ���@    ���@    ���@    �#�@    `�@     ��@     ��@    @��@    ���@    `��@    `V�@    `g�@    ��@    �C�@     ��@    `��@    @8�@    @��@    ���@    ���@    ��@    �O�@     ��@     ��@     ��@    @o�@    @��@    ���@     ��@     �@     �@     I�@    @@�@    �y�@    ���@     ��@     O�@    ���@    ���@     \�@     8�@    ���@     \�@     ��@    �l�@     �@     e�@     ��@    �P�@    �Z�@    ���@     ��@     1�@    �}�@     u�@     $�@     -�@     �@     �@     �@     
�@     �@     ��@     �@     g�@     �@     e�@     ��@     ��@     z�@     &�@     �@     Ȭ@     �@     @�@     ��@     ��@     0�@     ^�@     �@     �@     ��@     �@     �@     p�@     f�@     ~�@     ��@     Π@     X�@     l�@     l�@     �@     8�@     ��@     ̙@     4�@     l�@     d�@     Ԗ@     D�@     ��@     <�@     H�@     H�@     ��@     8�@     ��@     d�@     �@     ,�@     H�@     ��@     ��@     t�@     ؑ@     |�@     |�@     ��@     ܐ@     p�@     ��@     ��@     �@     p�@     ��@     0�@     ��@     x�@     ��@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     @�@      �@     ��@     @�@     �@     H�@     H�@     ��@     h�@     ��@     (�@     ��@     ��@     8�@     (�@      �@     ؁@     h�@     h�@     �@     ��@     8�@     p�@     `�@     ��@     (�@     �@     �}@     �~@     �|@     �~@     �@     P�@     @|@     �|@     �~@     �|@     `|@     �z@     �{@     �{@     �z@     {@     @x@     �x@     {@     `v@     �w@     �y@     `w@     v@     �w@     @w@     pu@     �u@     �t@     �t@     �v@     �u@     Ps@      s@     �t@     0s@     pv@     �r@     �t@     `t@      r@      q@     pq@     �r@     �s@      r@     �p@     �r@      p@     �p@     �q@     �o@     �o@     @q@     p@      p@     �p@     �o@     �p@     Pp@     �o@     �m@      m@     �n@      k@     �i@      k@     �i@     `h@     �k@     �i@      i@     �i@     `h@     �k@      m@      l@     `k@     �g@      f@     �f@      h@     �e@     �e@     �e@     �d@     �a@     �d@     �c@      e@     �c@     @c@     �c@     �e@      c@     �b@     �`@     `b@      a@     �_@      c@      `@     ��@     c�@     �c@     �g@     �c@     `d@      g@     `e@     `f@      c@     `e@     @h@      i@      m@     �k@     @k@      j@     �f@     �i@     `i@     �i@     �j@      j@     �g@     `d@     �l@     `j@     �g@     �j@     �j@     �p@     @o@     @l@     `o@     @m@     �o@     �m@      o@      o@      m@     �p@     @q@     Pq@     �p@     0p@      p@      p@     �q@     Pp@     0p@     `p@     �p@      s@      r@     �q@     �r@     �t@     �r@     �t@     �t@     �s@     r@     ps@     �t@     �t@     Pt@     `v@      y@     �v@     �w@      v@     �v@     �v@     �x@     y@     �w@     �y@     �z@     P{@     @{@     @z@     `z@     p}@     �|@     �z@     Py@     �}@      }@     �|@     �|@     P~@      �@     �@     0@     `@     p�@      ~@     �~@     ��@     �@     H�@     ��@     H�@     ��@     ��@     ��@     ��@     ��@     ȃ@     ��@     x�@     �@     ��@     ��@     @�@     ؅@     ��@     ��@     ��@     8�@     x�@     p�@     ��@     ��@     (�@      �@     `�@     h�@     ��@     �@     �@     ��@     ��@     ��@     ��@     ��@     H�@     ��@     ��@     p�@     �@     Ѝ@      �@     \�@     ��@     ��@      �@     ��@     �@     ̐@     d�@     0�@     (�@     ܑ@     ��@     x�@     ̔@     `�@     `�@     �@     ��@     �@     �@     \�@     ��@     �@     `�@     �@     ��@     ��@     ��@     �@     (�@     ��@     x�@     ̝@     t�@     ��@     h�@     ��@     П@     b�@     :�@     R�@     \�@     ��@     �@     $�@     �@     ��@     p�@     z�@     ̨@     �@     ,�@     ��@     V�@     �@     ��@     �@     ұ@     ��@     4�@     ;�@     ��@     a�@     q�@     �@     �@     �@      �@     u�@     ��@    ��@     ��@     :�@    �I�@    ��@     	�@     j�@    ���@    �)�@    �-�@     ��@    �M�@     ��@    �=�@     ��@    �K�@     ��@    �O�@    �E�@    ���@     `�@    @�@     ��@    @��@     Q�@    �)�@     ,�@    @��@     ��@    @��@     �@    ���@    @��@    @��@     7�@    @�@     *�@    ��@    `��@    �C�@    @��@     ��@    ���@    ���@    �r�@    @�@     ��@    ���@    `z�@     i�@    �O�@    @��@    ���@    ��@    ���@    �>�@     ��@    @��@    ���@    @V�@    ���@     ��@    @�@    @�@    ��@    @��@     6�@    @��@    @��@    ���@     c�@    ���@     ��@     ��@    @M�@    `��@    ���@     ��@     �@    ���@     @�@     ��@     �@     (�@     (�@     ��@     ��@     �@     $�@     ��@     ��@        
�
predictions*�	   @�0��   ��S @     ί@!  ���&@)8�x�F:@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���ڋ��vV�R9��T7����5�i}1�>h�'��f�ʜ�7
�O�ʗ�����Zr[v���ѩ�-߾E��a�Wܾ8K�ߝ�>�h���`�>f�ʜ�7
?>h�'�?�5�i}1?�T7��?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?ܔ�.�u�?��tM@�������:�              @      @      @      (@      .@     �@@     �F@      F@     �R@     @W@     �Y@      Z@      \@     �[@      Z@     @Z@     �X@     �[@     �`@      W@     @Z@      [@     @V@      W@     �T@     �N@     @S@      O@      N@      E@     �@@      C@     �B@      :@      =@      >@      5@      6@      ;@      0@      1@      (@      &@      "@      (@      "@      @      "@      @       @      @      @      @      @      @      @      @      @               @      �?       @               @               @               @      �?       @      �?              �?              �?              �?              �?              �?              �?              �?              �?              �?               @      �?      �?              �?      @       @      @      �?      �?      @       @      �?       @      @      @      @      @      �?      @      @      @      @       @      *@       @      @      ,@      @       @      &@      *@       @      .@      0@      ,@      3@      0@      <@      9@     �B@      ;@      ?@      D@     �@@     �B@     �B@      ?@      E@      A@      E@     �B@     �C@      B@      C@      F@     �@@      <@     �B@     �F@     �E@      D@     �A@      =@     �F@      >@      >@      2@      4@      1@      8@      *@      *@      "@      &@       @       @               @      @       @      �?              @      @              @       @               @      �?      �?       @               @      �?              �?        %��)�2      u�	�Mc���A!*�e

mean squared errorE�=

	r-squared�3R>
�L
states*�L	      �      @   �$[RA!.
�Z��)�b:��$A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             4�@     D�@     0�@     ��@     �@     d�@     ��@     ��@     ��@     &�@     ��@     b�@    ���@    ���@     ��@    ���@    ���@    �l�@    @��@    ���@    ���@    �
�@    ���@    ���@    ���@    `��@    ���@    ���@    ���@    @��@    `��@    @�@    ���@     �@     N�@     [�@    ���@     ��@    `��@    ���@    �f�@    ��@    `!�@    ��@    @��@     ��@     ��@    ���@    @��@    �H�@    �$�@     ��@    ���@    @��@    �n�@    @��@     �@     ��@    @o�@    �p�@    �~�@    ���@    ��@    �\�@    @��@     m�@    ���@     )�@     `�@    �`�@     ��@    �"�@    @*�@     ��@    ���@     ��@     �@     ��@    �?�@     g�@    �R�@     |�@     ��@    ���@    ���@    ���@     \�@     ��@     ��@    �9�@     ��@     �@     ׻@     U�@     ��@     U�@     �@     ��@     ��@     �@     ��@     ڲ@     B�@     ��@     Ȱ@     ��@     "�@     :�@     B�@     ҫ@     N�@     ��@     �@     t�@     F�@     ^�@     N�@     ��@     t�@     �@     P�@     ��@     �@     ��@     ��@     h�@     X�@     $�@     �@     ��@     ��@     �@     ��@     ��@     �@     ��@     �@     ��@     \�@     ��@     �@     ��@     p�@     �@     t�@     P�@     ��@     ؑ@     ��@     ,�@     �@      �@     p�@     ��@     �@     ��@     Ȍ@     `�@     ��@     P�@     ��@     p�@     ��@     ��@     �@     h�@     p�@     x�@     ��@     ��@     Ї@     ��@     x�@     @�@     ��@     ��@     h�@      �@     ��@     H�@     ��@     h�@     P�@     ��@     ��@     p�@     �@     X�@     Ȅ@     �@     ��@     ��@     P�@     ��@     h�@     �@      �@     ��@      �@     �@     �@     �}@     0~@     x�@     �y@     �}@     �{@     �y@     �~@     @z@     0{@      }@     �|@     �{@     �z@     �y@     �{@     P{@     �x@     �x@     �w@      y@     Pu@     �x@     Px@     �w@     �t@     pu@     Pu@     0u@     @v@     �u@     @v@     �r@     �s@     �t@     Pu@     @t@     0r@     Ps@      q@     @r@      r@     �q@     `q@     �n@      o@     @o@      q@     �o@     �l@     �n@     0q@     �l@      p@     @m@     �m@     `l@     `l@      o@     �k@     0q@      m@     �o@     �l@     �k@      m@     �i@     �k@     �j@     `i@     �m@     �l@      i@     �i@     �f@     �g@      i@     �g@     �h@      f@     `g@     �h@      d@      d@     `d@     `d@     �f@     `c@     @a@     �b@     `b@     `b@     `a@      f@     �b@     `c@     �a@     �c@     �a@      b@     �\@     ��@    ���@     �a@     �d@     �e@     �j@     @f@      h@      h@     �f@     �f@     �f@     �g@     �g@     @g@     �h@     �i@      h@     �g@     �g@      e@     �h@      i@     �i@     `i@     �k@     �i@     �h@      j@     �i@     �g@     �h@      k@     �m@      n@     @j@     �m@     @n@     �k@     �o@     �l@     �j@     �o@     pq@     pp@     �p@     �o@      q@     �o@     �p@     s@     �q@      s@     �s@     �r@     Ps@     pr@     �r@     �q@     pt@     �s@     �t@     �s@      u@     Pt@      s@     �v@     0u@     �u@     �v@     �w@      x@     �x@      z@     �w@      v@      y@     �{@     P{@     z@     �w@     �{@     �x@     Pz@     0z@     P}@     �{@     �|@     �|@     @{@     p@     p�@      ~@     p}@      }@     @@     �@     @@     �~@     �@     X�@     ��@     p�@     p�@     ��@     �@     `�@     �@     �@     Ѓ@     ��@     p�@     ��@     Є@     ��@     h�@     x�@     ��@     ��@     x�@     ��@     h�@     ��@     ��@     0�@     ��@     ��@     ��@     x�@     ��@     ��@     P�@     `�@     �@     ��@     ��@     8�@     �@     X�@     8�@     `�@     ��@     ��@     ��@     x�@     ؎@     X�@     �@     T�@     \�@     (�@     `�@     ��@      �@     ��@     ��@     ��@     ��@     ��@     ��@     \�@     �@     ��@     Ԗ@     d�@     8�@     8�@     �@     $�@     �@     ��@     ��@     ��@     �@     P�@     P�@     �@     (�@     ��@     �@     8�@     �@     4�@     ��@     ��@     b�@     ~�@     (�@     ��@     T�@     ԧ@     ��@     0�@     <�@     N�@     Į@     b�@     �@     α@     �@     �@     v�@     D�@     x�@     *�@     �@     �@     ��@     �@     �@     ?�@     ��@    �!�@     ��@    ���@    �Y�@     �@    ���@    ���@    ���@     ��@     D�@    ���@    ��@    ���@    ���@     .�@    ���@    ���@    �~�@    ���@    �5�@     ��@    �|�@    ��@    ���@    ���@    �"�@    ���@    @��@    @s�@     ��@    @��@    @��@     ��@    ���@    ���@    ���@    �4�@    ��@     t�@    ���@    �X�@     �@     S�@    �r�@    �h�@     �@    ���@    ��@    `A�@     �@    ���@    @$�@    @��@    ���@     7�@    @��@    @��@    @q�@    @-�@    ��@     @�@    @g�@     O�@    @��@    ���@    �)�@    ���@    @H�@     	�@     �@    ���@    @��@    �O�@    @��@    ���@    �O�@     V�@    @��@    @��@    �E�@    ���@     #�@     ��@     ��@     4�@     z�@     ©@     Ƣ@     ,�@     ��@     ̖@     ��@        
�
predictions*�	   �gڻ�    ��@     ί@!  pF� �)8.��@@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"���d�r�x?�x����Zr[v��I��P=��pz�w�7��})�l a��ߊ4F����~]�[Ӿjqs&\�ѾK+�E���>jqs&\��>a�Ϭ(�>8K�ߝ�>��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?�iZ�?+�;$�?cI���?�P�1���?3?��|�?S�Fi��?ܔ�.�u�?�Š)U	@u�rʭ�@�������:�               @      @      (@      9@      <@      F@     �I@      O@     �Q@     �R@     �X@     �[@     �[@     �W@     @]@     �]@      [@     �\@     @W@     @Y@     �Z@     �T@     �Y@     �V@     �P@     �O@      N@      J@     �M@      F@     �D@      F@     �C@     �D@      E@      7@      7@      8@      ;@      3@      *@      6@      8@      @      &@      &@      "@      "@      @      @      @      @      @      @      @      @      @      @      @      @       @       @      �?      @      @               @              �?              �?              �?              �?      �?              �?              �?              �?              �?              �?               @               @      �?               @              �?      �?      @      �?       @       @      �?       @      @      �?      @      @      @      @      @      @      @      @      @      @      @      @      "@       @      .@      *@      .@      $@      $@      0@      ,@      .@      :@      8@      2@      A@      <@      :@      8@     �A@     �A@     �A@      8@      ?@      >@      B@     �D@     �C@      C@     �A@      B@     �C@      =@      @@      B@      A@      A@      A@      B@      B@      ?@     �@@      6@      3@      6@      7@      8@      *@      1@       @      @      "@      @       @      @      @      @       @      �?       @       @      @              �?      �?      �?      �?      @               @       @      �?      �?              �?              �?        U��R�3      �=%�	���c���A"*�g

mean squared error�}+=

	r-squared0� >
�L
states*�L	      �      @   �$[RA!S�lT�}��)-�=[��%A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             &�@     ��@     ��@     |�@     b�@     �@     ʯ@     �@     U�@     y�@     }�@    �N�@    @��@     K�@     ��@    `��@    ���@    �e�@    �?�@    ���@    @
�@    @ �@     ��@    ���@     ��@    �S�@    �:�@    `�@    ��@     �@    @�@    ��@    ��@     f�@    �|�@     u�@     ��@     �@    @�@    ��@    `��@    @��@    `��@     _�@    @$�@     4�@    �6�@    ��@    �0�@    ��@     ��@     C�@    � �@    @]�@    ���@    �&�@    �u�@    @��@     ��@    @@�@    �A�@    @��@    @��@     �@     ��@     b�@    ���@     ��@    ���@    @0�@    ���@    �(�@    @��@    �j�@    ���@     �@    �	�@     ��@     ��@     ��@    ���@    ���@    �I�@    �<�@    �:�@     ��@    ���@     ��@     M�@    �<�@     0�@     ��@     6�@     #�@     ��@     '�@     "�@     X�@     ��@     -�@     ִ@     �@     Q�@     ��@     n�@     �@     K�@     ��@     �@     J�@     p�@     ��@     �@     ��@     r�@      �@     ��@     �@     d�@     ��@     V�@     �@     f�@     �@     0�@     �@     ��@     ��@     ��@     `�@     4�@     ��@     L�@     ؙ@     ��@     ,�@     �@     ��@     ��@     P�@     ��@     �@     ��@     ��@     Ȓ@     d�@     (�@      �@     ��@     �@     X�@     x�@     ��@     ��@     0�@     `�@     (�@     ��@     0�@     ��@     h�@     �@     ��@     `�@     ��@     H�@     ��@     ��@     ؉@     �@     ��@     h�@     �@     �@     X�@     X�@     Ȇ@     X�@     ��@     (�@     ��@     ��@     X�@     (�@     ��@     �@     Ѓ@     x�@     @�@     ��@     ��@     ��@     p�@     �@     X�@      �@     P�@     �@     x�@     0�@     @}@     �|@     ~@     �~@     �|@     �}@     x�@     �~@      ~@     �{@     �z@     `}@     @{@     �z@     0z@     �y@     �z@      z@     Pz@     �w@     z@     �w@     pz@      v@     �v@     @w@     `u@      u@     �s@     �s@     0w@     @t@     �r@      s@     �s@     �s@     `r@     �s@     �r@     Pt@     �t@     �p@     �q@     �s@     q@     �q@     �n@     �p@     �o@      p@      n@     �q@     `q@     �o@     �k@     @p@     `m@     �l@     �n@     `n@     �k@     �n@     �n@      l@     �j@      k@      j@     �j@      i@     �i@      i@     �i@     �i@     �f@      i@     `h@     `i@     �h@     `h@     �f@     `f@     �i@     �f@      e@     @g@     `h@     �g@      e@     @e@     �d@      c@     �c@     �a@     �e@     �b@     `a@     @c@     �_@     �a@     @d@     �a@     @^@    ��@     �@     �c@     @c@     �d@     `c@     @g@      g@      g@     @e@     �g@     �f@      k@     `f@     �d@      g@     �g@     �j@     `f@     �i@     �g@     `h@      h@     @h@      i@     �f@     �l@     �k@     �j@     �j@      j@     �h@      k@     �m@     @l@      l@     �o@     �j@     �m@     �n@      k@     �p@     `q@     �j@     �q@      p@     �p@     �p@      p@     0q@     @p@     pp@     Pr@     �r@     0s@     r@     �s@     t@     0r@     �t@     �t@     `u@     �r@     �s@     `u@     �s@     �u@     �t@     t@     @v@      v@     v@     y@      y@     `y@     �v@     �y@     y@     y@     �z@     �z@     0z@     �{@     0{@      }@      {@     P}@      |@     �{@     �}@     �~@     �|@      �@     P~@     �~@     �~@     P�@     �}@     �}@      �@     H�@     8�@     ��@     x�@     p�@     ��@     ��@     h�@     ��@     ��@     `�@     ��@     �@     0�@     Є@     ��@     p�@     ��@     P�@     Є@     `�@     �@     І@     p�@     ��@     P�@     H�@     @�@     �@     Ȉ@     ��@     0�@     ��@     ��@     8�@      �@     `�@     x�@     �@     �@     (�@     @�@     @�@     ��@     ,�@     Ȑ@      �@     ��@     H�@     ��@     ��@     4�@     <�@     ԑ@     ��@     T�@     ��@     4�@     ,�@     4�@     X�@     ��@     ��@     ��@     h�@     �@     �@     8�@     ̙@     P�@     L�@     �@     `�@      �@     �@     ��@     D�@     z�@     �@     J�@     
�@     Т@     ��@     B�@     �@     F�@     ��@     h�@      �@     �@     �@     ��@     ��@     $�@     ��@     k�@     �@     �@     ,�@     ��@     ��@     g�@     �@     ��@     z�@     5�@     ��@     ��@     $�@     ��@     ��@     I�@    �d�@     �@    ���@    �<�@     ��@    ��@     ��@    ��@    �Z�@     7�@    ���@    ��@    �,�@    ���@    �Y�@    �t�@    �x�@    @v�@    ���@    �R�@    �*�@    @��@     `�@    ���@    @��@    �+�@    @��@    @��@     ��@    @b�@    @R�@    �>�@    �O�@    �b�@    @;�@    �t�@    �W�@    �2�@    @��@    ��@    ��@    `(�@    `C�@    ��@     
�@    `��@    �Z�@    ��@    �}�@    ���@     f�@    �R�@    ���@    ���@    @��@    �g�@    @C�@     ��@     u�@    ���@    ���@     [�@     {�@    �3�@    �"�@    ��@     ��@    ���@    ��@     \�@    @E�@    ���@     ��@    @��@    @#�@    �"�@    @>�@     ��@    @R�@    @��@     h�@    �R�@     ��@     ��@     {�@     ح@     \�@     ��@     |�@     <�@     �@     ��@        
�
predictions*�	    �a��   `���?     ί@!  x59\<@)�,�im�)@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�f�ʜ�7
�������FF�G �>�?�s�����Zr[v��I��P=��})�l a��ߊ4F��iD*L�پ�_�T�l׾�[�=�k�>��~���>�����>
�/eq
�>jqs&\��>��~]�[�>�uE����>�f����>�ߊ4F��>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?1��a˲?6�]��?>h�'�?x?�x�?��d�r?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�               @      �?      @      @      *@      ,@      9@      ?@      =@     �E@      A@      L@      C@      M@     �I@     �L@     �Q@     @P@     �L@     �P@     �K@      M@      O@     @Q@      I@     �D@     �E@      O@      E@     �E@     �E@     �B@      F@      <@      C@      5@      A@      :@      7@      ;@      .@      2@      .@      $@      $@      1@      &@      .@       @      @      $@      @      &@      @      @      @      @      @      @       @      @       @      �?      @      �?      �?               @               @       @       @      @              �?              �?              �?               @              �?              �?              �?              �?              �?              �?              �?      �?      �?      �?              �?      �?              �?              �?       @              �?      �?      �?      �?      @      @      �?      @      @       @      �?               @       @      �?      @       @      @      @      @      @      @      @      *@      "@       @      @      @      2@      3@      ,@      0@      1@      9@      5@      :@      6@      A@     �B@     �G@      K@      K@     �Q@      M@      R@      Q@     @T@     @V@     @W@     �X@      V@     �W@     �S@     �Q@     �P@     �S@      R@     �I@     �C@     �F@     �J@      E@      9@      ;@      ;@      8@      >@      5@      8@      :@      5@      3@      ,@      @      3@      (@      (@      @       @      @      @      @      @      @      @      @      �?      �?       @      �?               @      �?               @              �?       @      �?        �ﳳ�2      ���C	K<Dd���A#*�e

mean squared error�Y$=

	r-squared��%>
�L
states*�L	      �   ���@   �$[RA!ң�~�[��)J!~�&A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ܯ@     ԗ@     p�@     (�@     <�@     �@     ��@     ߲@     �@     Y�@    �
�@    �z�@     ��@    ���@    �e�@    �;�@    ���@    �X�@    ���@    ���@     ��@    @,�@    �p�@    `��@    ���@    ���@    ���@    �f�@    �|�@     R�@     W�@    `b�@    ��@    `��@     ��@    ���@    ���@     /�@    @c�@    `��@    �o�@     \�@    �5�@    ���@    @��@    ���@     ��@     ��@    �2�@     .�@    �D�@    ���@    @9�@    @n�@    @��@    ���@    ��@    ���@    @��@    �<�@    �Y�@    @E�@    @��@    � �@     )�@    �i�@    ���@     ��@     7�@    @�@    ���@    ���@    �~�@    ��@     ��@    �)�@    ���@     ��@     ?�@    ���@    ���@    �?�@    ���@    ���@    ���@     ��@    ���@     ]�@     j�@     =�@     %�@     Ľ@     ��@     |�@     -�@     �@     .�@     Ѷ@     ��@     =�@     n�@     #�@     �@     ��@     ��@     ��@     ��@     ��@     ^�@     ��@     �@     z�@     ��@     n�@     ��@     ̦@     z�@     ��@     У@     Ф@     آ@     ��@     �@     `�@     n�@     �@     ��@     ��@     $�@     Ԝ@     p�@     4�@     �@     �@     ��@     <�@     $�@     ��@     ��@     ,�@     (�@     4�@     `�@     ��@     ��@     ܒ@     @�@     4�@     `�@     �@     ��@     �@     ��@     ��@     d�@     �@     �@     p�@     �@     `�@     `�@     ��@     ��@     ��@     ��@     �@     ��@     0�@     `�@     ȇ@     ��@      �@     ��@     H�@     ��@     p�@     ��@     Ȉ@     @�@     ��@     �@     X�@      �@     ��@     `�@     �@      �@     ��@     ��@      �@      �@     ��@     �@     ��@     8�@     ��@     Ȃ@     ��@     ��@     `�@     @�@     ��@     �@     @     0�@     (�@     `�@     x�@     @     }@     �@     `}@      |@     �{@     �z@      {@     �|@     �z@     {@     �}@      z@      x@     �w@     �z@     Pz@     @v@     �v@     px@     �x@     @w@     @w@     �q@     �t@     �w@     �v@     �u@     `s@      v@     �s@     �t@     �t@     �r@     �r@     pp@     �p@     �p@     Ps@     pp@     �q@     Pp@     q@     �o@      o@     �q@      n@     @o@      o@      n@     �l@     �m@      n@     @n@     �l@     �m@     @l@     �i@     �m@     �m@      o@     �h@     �k@      k@     �k@     @i@      j@     �h@     �f@      i@      i@      g@     `f@     `m@     �g@     `f@     @f@     �d@     �e@     �f@     �d@     �f@      j@     `c@     `d@      f@      e@     `b@      b@     �e@     �a@     �d@     `a@     �a@     V�@     r�@     �d@     �d@     @c@     �f@      e@     �f@     @f@     @h@     �g@     �f@     �i@      g@     �h@     �i@     �i@     �h@      j@     �h@     @g@     �h@     @j@     �j@     �k@     �i@     �k@     �n@      o@     �l@     �k@      p@     �l@     �n@     �p@     �o@     �o@      j@      p@     �n@     `q@     �m@     �p@     Ps@     �o@     �q@     �r@     q@      r@     q@     �q@     �s@     �q@     �t@     �q@     Ps@      s@     �s@      u@      u@     0v@     �u@      u@     �u@     �v@     pw@     `x@     v@     v@     pw@     �x@     @{@     Py@     �{@      x@     �z@      {@     `{@     �{@     �z@     |@     @|@     `{@     {@     �~@     �@     �~@     �~@     �}@     P~@     p�@     p�@     �~@     �@     �~@     x�@     h�@     X�@     ��@     �@     ��@     �@     p�@     Ȃ@     ��@     x�@     Ђ@     ��@     p�@     ��@     ��@     ��@     X�@     0�@      �@     h�@     x�@     �@     ��@     �@     ��@     ��@     �@     @�@     x�@     ��@     8�@     @�@     ��@     �@     H�@     �@     ��@     �@      �@     �@     h�@     h�@     ��@     ��@     �@     <�@     ؐ@     ؏@     ��@     �@     ��@     �@     ��@     ��@     D�@     \�@     ��@     `�@     l�@     ��@     `�@      �@     �@     ��@     Ȗ@     ̖@     �@     ̗@     ��@     ��@     d�@     |�@     ��@     0�@     �@     �@     ��@     �@      �@     ğ@     P�@     �@     ��@     ֡@     "�@     ��@     V�@     �@     �@     Υ@     ¦@     ��@     H�@     @�@     4�@     ��@     f�@     ��@     H�@     +�@     y�@     ߱@     �@     ��@     ��@     �@     ִ@     W�@     @�@     ��@     ϸ@     ��@     ��@     �@     �@     �@     ^�@    �q�@    �[�@     ��@     ~�@    �z�@    �D�@     ��@    ���@     ��@     ��@     _�@     ��@    ���@    ���@     *�@    ���@     }�@    ���@    �b�@    @��@    @2�@    ��@    �j�@    �c�@    @��@     ��@     r�@    �E�@    @#�@     �@     ��@    ���@    @��@    ���@    ���@     �@    @��@    �M�@    `��@     ��@    `
�@    ��@    ��@    @��@    `y�@    �;�@    ���@     ��@    ��@    ���@    ��@    �F�@    ���@    ���@     w�@     ��@    �*�@     ]�@     c�@    ���@     ��@    ��@    ���@    ���@    ���@    ���@     ��@    ��@    ���@     U�@    �b�@     x�@    ���@     ��@    ���@    @H�@    �h�@    @Z�@    �3�@     ��@    ���@     @�@     .�@     p�@     x�@     P�@     �@     ��@     �@     l�@     h�@        
�
predictions*�	   @ڻ�   ���@     ί@!  e[B@)��6;�D@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ���d�r�x?�x��>h�'���FF�G �>�?�s���pz�w�7��})�l a�X$�z��
�}������~���>�XQ��>>h�'�?x?�x�?��d�r?�5�i}1?�T7��?��ڋ?�.�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?w`<f@�6v��@!��v�@زv�5f@�������:�              @      @      *@      .@      6@      ;@      ?@     �B@     �J@      F@     @P@     �O@      Q@     �N@      R@     �O@     �N@     @R@     �N@      J@      R@      P@     �N@     �I@     �Q@      M@      H@      9@      D@      F@     �C@     �@@     �D@      >@      9@      6@      0@      6@      5@      0@      2@      2@      (@      ,@      "@      @      $@      "@      @      $@      @      @      @      @      @      @       @       @       @       @      @               @      �?      �?               @      �?       @               @      @      �?      �?              �?      �?               @              �?              �?              �?              �?      �?       @      �?              �?              �?               @       @              �?      �?       @      @       @       @      @      @      @       @      @      $@      @      �?       @      &@      "@      &@      @      .@      (@      1@      0@      5@      5@      =@      5@      8@      7@      6@      =@      D@      C@     �A@     �H@      H@      J@     �N@      G@     �T@     �S@     �T@     �T@     �Q@     @S@     �Q@     �Q@     �N@     �P@     �N@      L@      N@     �J@     �H@      D@     �E@     �B@     �E@     �@@      ;@      6@      ;@      <@      0@      4@      1@      0@      (@      .@      ,@      @      @      "@      @      @      @      @      @      �?       @               @              �?       @      �?       @      �?              �?              �?              �?        �`b3      ��v	I��d���A$*�f

mean squared error�L=

	r-squared4|E>
�L
states*�L	      �    ��@   �$[RA!�J��,��)���;'A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             |�@     ș@     <�@     (�@     @�@     �@     �@     ��@     �@    ��@     G�@    ���@     ��@    ���@    �(�@    ���@    �
�@    @:�@    ���@    @�@    ��@    @��@    @�@    `+�@    �)�@     �@    �E�@     ��@    ��@     ��@     ��@    ���@     ��@    @��@     ��@    ��@    ��@    `0�@    �>�@     @�@    ��@     ��@    ���@     ��@    `��@     ��@    �E�@     0�@    ���@     ��@    �h�@     ��@    @��@    �w�@    ��@    @��@    �i�@    @��@    ���@    ��@     S�@    ���@     ��@    ��@    �u�@    �R�@     ��@    ���@    ��@     s�@    �&�@    �h�@     ��@    ���@    ���@     _�@    ���@     ��@     |�@     ��@     ��@     ��@     �@    ��@     ��@    �`�@     ��@     ��@    ��@     ��@     ��@     ��@     �@     u�@     ��@     ��@     �@     ո@     ��@     ;�@     =�@     t�@     {�@     ��@     ��@     �@     ��@     ��@     F�@     ҭ@     @�@     ,�@     ��@     ��@      �@     ��@     �@     P�@     ��@     f�@     ��@     �@     8�@     ��@     ��@     �@     R�@     x�@     `�@     �@     �@     Ĝ@     (�@     ��@     ��@     0�@     ��@     <�@     ��@     H�@     �@     ��@     ��@     ��@     ��@      �@     Д@     ��@     <�@     �@     p�@     t�@     ��@     D�@     ��@     D�@     P�@     h�@     (�@     H�@     h�@     ��@     ��@     ��@     `�@     ��@     �@     X�@     Њ@     ��@     ��@     �@     x�@     (�@      �@     x�@     ��@     ؈@      �@     x�@     ��@     ��@     �@     ��@     ��@     Є@     h�@     P�@     Ȅ@     ��@     p�@     X�@     ��@     H�@     ��@      �@     @�@     �@     X�@     Ȁ@     �~@     �@     8�@     �@     ��@     �@     �@     �@     �@     P~@     �|@     0�@     @}@     �{@     `|@     @|@      {@     {@     `z@     �x@     0y@     `y@     �y@     �w@     `y@     `w@     �w@     �x@     �v@     �u@     �u@     pv@     �s@     0u@     �u@     Pr@     �s@     `t@     �t@     �t@     �s@     �u@     �u@     �t@     @r@     �r@     �q@     @r@     �r@     @s@     �r@     0r@     �q@     �p@     �p@     @o@      o@     �o@     �o@     �m@     @p@     �m@     �l@     �j@     @l@     �k@     �k@      n@      j@     �i@     �h@     @m@     `n@     �f@      k@      h@     �k@     �h@     @l@     @m@      h@      g@      g@     `g@     �i@     �h@     �e@      i@      f@     �e@      d@     �e@     �c@     `c@     @c@     �`@     �b@     `d@     @a@      a@     `e@     �e@     ��@     ��@     �h@     �c@     �d@     `g@     �f@     �f@     �g@     �i@      i@     �e@     �g@      i@     `k@     �g@     �k@     `h@     �k@     �k@     `i@      k@     �k@     `h@     �j@     `i@     �j@     @l@     �l@     �m@     `n@     �m@     `k@     �o@     @o@     �q@      q@     pp@     �p@     `k@     �p@      q@      o@     �o@     @r@      n@      o@     �r@     �p@     �q@     �p@      t@     �r@     �s@     Pt@     Pr@     Ps@     Pv@     �q@     �s@     �r@     u@      t@     �v@     `v@     �v@     pu@     @v@     �x@      w@     �x@     �w@     �x@     P|@     �x@     �|@     Pz@     @{@     �y@     �{@     �}@     0z@     �{@     (�@     �}@     �}@     �}@      |@     X�@     8�@     p�@     �@     �~@     �@     ��@     Ȁ@     ��@     8�@     P�@     ��@     ��@     ؁@     ��@     �@     ��@     ��@     Ѐ@     ��@     p�@     x�@     ��@     �@     �@     p�@     Ѕ@     І@     �@     ��@     ��@     0�@     ��@      �@     `�@     �@     ��@     ��@      �@     `�@     0�@     ��@      �@     ��@     p�@     0�@     (�@     ��@     `�@     �@     ��@     X�@     �@     Ȏ@     t�@     Ў@     ��@     ��@     ��@     0�@     �@     �@     p�@     ��@     ��@     (�@     d�@     ؓ@     ��@     ��@     ��@     �@     h�@     ��@     �@     ��@     �@     �@     x�@     8�@     |�@     ��@     ��@     ,�@     ,�@     D�@     ̟@     <�@     ��@     H�@     ��@     Ȣ@     �@     أ@     �@     ޤ@     z�@     ��@     �@     ��@     �@     ª@     ��@     ��@     8�@     ��@     Я@     !�@     ��@     I�@      �@     ӳ@     ��@     �@     ��@     ̷@     �@     ��@     �@     ڻ@     �@     ۽@    �"�@    ���@    �(�@    ���@     z�@     ��@     ��@    ���@    �?�@    ���@     ��@     ��@    ���@    ��@    �!�@     %�@    ���@     ��@    ���@    ���@     ��@    ���@    ��@    ���@     �@     ��@    �(�@     :�@     ��@    @U�@     
�@    ���@    ��@    �"�@     ��@    @o�@    @o�@     ��@     6�@     T�@     /�@     ��@    ��@     \�@    �x�@    `Q�@     1�@    �)�@     ��@    �,�@    @��@    �<�@     ��@    �~�@    �A�@    ��@    ���@    ���@    �n�@    �a�@    @W�@     ��@    @1�@    ���@    @��@    ���@    ���@    ���@    @W�@     ��@    ��@    @��@    �v�@    �N�@     k�@    �t�@    @��@    �;�@    �2�@    @0�@      �@    ���@     ��@     �@     ��@     ʷ@     /�@     ��@     ��@     �@     ��@     H�@     p�@     6�@        
�
predictions*�	   ���ÿ   �'�@     ί@!  �ш8@�)�<��<@2�yD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7���x?�x��>h�'��1��a˲���[��>�?�s���O�ʗ���pz�w�7��})�l a�f����>��(���>a�Ϭ(�>8K�ߝ�>�h���`�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?uo�p�?2g�G�A�?������?�iZ�?cI���?�P�1���?�DK��@{2�.��@�������:�              �?      �?              @      @      ,@      9@      @@      F@      O@     �R@      S@     @X@     �U@      Y@      ]@     @\@     �_@     @Y@     �]@      \@     �Y@     @W@     @X@     @R@     @U@     �T@     �U@     �W@     �R@     @Q@     �F@     �J@     �D@     �E@      ?@      D@     �B@     �@@      8@      :@      9@      9@      >@      3@      3@      $@      ,@      0@      *@      "@      (@      "@      @       @       @      @      �?       @      &@      @      @      @      @       @       @      �?      @               @      �?              @      @              �?      �?      �?      �?              �?              @              �?              �?              �?              �?      �?              �?              �?       @       @              �?              �?      �?               @              �?               @      @      �?       @       @       @      @      @      @      �?      @      @      @      @      @      @      @      @      @       @      $@      @      "@      &@      &@       @      "@      ,@      3@      1@      6@      :@      4@      "@     �B@      A@      7@      <@      A@      @@      B@      D@      A@     �@@     �A@     �I@     �C@      7@      <@     �A@      <@      <@      ?@      @@      9@      5@      1@      3@      9@      4@      ;@      *@      0@      "@      $@      .@      @      &@      $@      ,@      @      @      @      @      @      @       @              @      �?              �?      �?       @              @      �?      �?              �?              �?        #�ȜR3      ��	Lk;e���A%*�f

mean squared error]:&=

	r-squaredl>
�L
states*�L	      �   `��@   �$[RA!���Cs���)�櫷��'A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             6�@     ؙ@     ܚ@     ��@     
�@     �@     ��@     �@     ��@    ���@    �n�@     ��@    `��@    @r�@     ~�@    �	�@    @-�@     v�@     ��@    �/�@     �@     ��@    @��@     ��@    `�@    `��@    `��@    `��@    �T�@    `.�@     I�@    ��@    @1�@    `�@     d�@    �N�@    ���@    ���@    �!�@    �p�@     E�@     "�@    �,�@    ���@    ���@    @��@    ���@     e�@    �B�@    ���@    ��@     ��@     9�@    ���@    @K�@    ���@    ���@    ���@    �i�@    �s�@    ���@    �6�@    @D�@    @S�@    ���@    �)�@    ���@    @V�@    ���@    ��@     w�@    �v�@    ���@    ���@    ���@    ���@     p�@     i�@    �x�@    �T�@     ��@    �k�@     ��@     /�@     :�@     T�@     �@     ��@     ��@    ���@    �P�@     �@     ��@     �@     �@     �@     3�@     ��@     ��@     }�@     ��@     �@     ��@     j�@     ��@     *�@     w�@     ��@     ��@     ��@     H�@     ��@     t�@     ��@     �@     �@     `�@     �@     f�@     ^�@      �@     ��@     �@     �@     |�@     �@     ��@     ��@     z�@     �@     (�@     ��@     ؛@     �@     (�@     ��@     ��@     ��@     8�@     �@     ,�@     �@     Е@     �@     �@     x�@     \�@     ��@     ,�@     �@     ̑@     ��@     ��@     �@     x�@     P�@     ��@     �@     x�@     @�@     P�@     ��@     ،@     ��@     �@     ��@     Ȉ@     ��@     ��@     ��@     ��@     (�@     ��@     ��@     ��@     ��@     X�@     @�@     ��@      �@     ��@     ��@     (�@     ��@      �@     ��@     ��@     ��@     ȃ@     ��@     x�@     Ѕ@     x�@     x�@     �@     ��@     P�@     �@     ��@     0�@      @     �~@     �@     �@     ��@     �~@     �@      �@     ��@     �~@     �}@      {@     �|@     P}@     P|@     @~@      z@      |@     �}@     �y@     0}@      y@     `y@     �y@     0x@     �v@     �v@     pt@     Pw@     �y@     �v@     �t@      v@     pv@     �t@     �u@     �t@      s@     �t@     `s@     �r@     0s@      s@     Pr@     0s@     �r@     �s@      s@      n@     pr@      p@      r@     Pp@     @o@     �q@     �p@     Pq@     @o@     `o@     �p@     �k@     �o@     0p@     �n@     �l@     �l@     �m@     �l@      k@     �j@     �l@     �l@     �k@     �m@     �n@     �j@     �k@     `i@     �g@     �f@     �g@     `f@      e@     `g@     �e@     �d@      e@      h@      d@     �b@     �e@     `d@     @c@     �e@     �a@     �a@     `d@     `e@     �e@     �f@      e@     �`@    �A�@    ��@     �e@     �h@     �d@     �g@     �f@      f@     �i@     �h@      e@      h@     �g@     �j@     `g@     �g@      j@      i@      j@     �j@     �h@     �k@     @j@     @j@      j@     �l@     @m@     �n@     �k@     0p@     �n@     �m@      p@     �m@     �o@     @o@      o@     �q@     Pp@     Pp@     �o@     `q@     `p@     �r@      r@     pq@     �r@     �p@     �q@      t@     �r@      r@      s@     �s@     @s@     �t@     Pt@     @s@     @t@     �s@     `u@     �t@      v@     v@     pu@      v@     �u@     `w@      y@     �y@     �y@     �x@     �x@     �x@      z@     0y@     �x@     �{@     p{@     0z@     �x@     �~@     �@     P~@     `~@     �~@     �|@     �}@     �|@     �~@      }@     @     �@     �}@     �}@     �@     X�@     8�@     Ё@     X�@     ��@     0�@     ��@     ��@     P�@     �@     ��@     x�@     Є@     ��@     ��@     Ȅ@     ��@     ؃@     �@     �@     h�@     �@     (�@     (�@     ؆@     �@     x�@     (�@     ��@     ��@     ��@     H�@     `�@     X�@     X�@     (�@     x�@     h�@     P�@     @�@     X�@     x�@      �@     ��@     ��@     �@     ��@     (�@     4�@     ��@     d�@     t�@     Ȑ@     D�@     @�@     ��@     x�@     �@     �@     Ĕ@     �@     ��@     �@     �@     �@     ��@     ��@      �@     d�@     ؙ@     ܚ@     �@     ��@     8�@     ��@     ,�@     �@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     �@     �@     "�@     6�@     �@     ��@     ֧@     6�@     �@     p�@     ̬@     ��@     ҭ@     L�@     ��@     
�@     k�@     C�@     $�@     ߳@     ��@     �@     A�@     ��@     r�@     ù@     ˺@     Ļ@     ��@     (�@      �@     *�@    ���@    �^�@     F�@    ���@     ��@    �Y�@    ���@     o�@    ���@     q�@     O�@    ���@    � �@    ��@     ��@    ���@     ��@     >�@     ��@    �B�@    ��@    ���@    �.�@    @��@    �H�@    �	�@    �[�@     ~�@    �P�@    ���@    @��@    ���@    �Z�@     ��@    �{�@     ��@     8�@    ��@     ��@    ���@    `�@     �@    ��@    ���@     ��@    @n�@     K�@    �i�@    ��@    ���@    �'�@    ���@     \�@    ��@    ��@    @��@     ��@     �@    �d�@    @��@    ���@    �;�@     ��@    @?�@    @e�@     ��@     P�@    @8�@     ��@     ��@     
�@    ���@    @p�@    � �@    ���@    ���@    �K�@     ��@    ���@    ��@    ��@    �5�@     r�@     �@     9�@     �@     ��@     ��@     �@     ę@     ��@     ��@        
�
predictions*�	   �����   �H @     ί@!  	���B@)*����\@@2�!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&��[^:��"��S�F !��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��f�ʜ�7
�������FF�G �>�?�s����ߊ4F��h���`�K+�E��Ͼ['�?�;��~]�[�>��>M|K�>�uE����>�f����>})�l a�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?+�;$�?cI���?!��v�@زv�5f@�������:�              �?               @       @      @      @      (@      .@      1@      A@      A@      F@      D@     �K@     @Q@     �H@      P@     �H@      M@      J@      G@      N@     �K@     �K@      N@     �M@     �H@     �L@      J@      M@      D@     �H@     �@@     �@@      >@      ;@      ;@      6@      6@      5@      4@      0@      .@      *@      .@      ,@       @      ,@      *@      @      (@      "@      $@      @      @      @       @      @      @       @       @      �?      @      @       @      @      @       @       @              �?              �?              �?      �?       @      �?      �?              �?              �?              �?              �?              �?              �?      �?              �?              �?      �?              �?       @              �?      �?      �?      �?      �?       @              @       @       @      @      @              @      @       @      @      @       @      @       @      @      $@      @      @      "@       @      ,@      (@      (@      0@      .@      ;@      2@      >@      A@      C@      @@      G@      N@     �G@     @P@      K@     @P@     @P@     @U@     �W@     �W@     �T@     �P@     �W@     @T@     �W@     �Q@     �Q@     �N@     �P@     @P@      I@     �L@      I@     �A@     �H@      @@     �B@      <@      =@      >@      8@      ,@      1@      4@      *@      7@      *@      ,@      $@      @      "@      @      @      @              @      �?               @       @       @      �?               @       @       @              �?              �?        �0oBr3      �i	@��e���A&*�f

mean squared error>�0=

	r-squaredx{�=
�L
states*�L	      �   ���@   �$[RA!�>b@��)�,��(A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             E�@     ğ@      �@     ��@     ��@     �@     ;�@     ]�@     y�@     ;�@     ��@    @!�@    ���@    ���@    ���@    ���@     R�@    ���@    @��@     A�@    @��@    ��@    � �@    �D�@     p�@    � �@     �@    @��@    ���@    ���@    ���@     ��@    @b�@    �v�@     ��@    ���@     �@    @M�@    �
�@    �.�@    �*�@    ���@    `��@    ��@    ���@    @Z�@    �F�@    ��@    `�@    ��@    @��@    �G�@    ��@    @&�@    @��@    ���@    ���@    @��@    ��@    �H�@    ���@    @��@    ���@     \�@     M�@    �q�@    @��@    ��@     ��@    ���@    ���@    @�@     >�@     ��@    ���@     V�@     ��@    ���@    �O�@     @�@      �@    �n�@    �R�@     ��@     �@    �@�@     ��@    ���@    ���@    ���@     %�@     K�@     D�@     ̼@     ��@     �@     \�@     V�@     Ƿ@     ��@     !�@     �@     �@     �@     u�@     ��@     �@     ��@     �@     �@     ��@     $�@     ȭ@     T�@     �@     ة@     ,�@     (�@     b�@     ��@     �@     ��@     Ƥ@     ��@     ~�@     
�@     ��@     Р@     ��@     �@     D�@     d�@     �@     ��@     `�@     ��@     @�@     p�@     ��@     ��@     ��@     $�@     `�@     X�@     D�@     ��@     ��@     l�@     �@     H�@     <�@     �@     �@     đ@     ��@     ܐ@     �@     А@     �@     �@     ؐ@     ��@     ��@     �@     Ȋ@     @�@     Ћ@      �@     Ќ@     8�@     ��@     0�@     ��@     0�@     �@     ؈@     ��@     h�@     ��@     ��@     ��@     �@      �@     ��@     ؅@     ��@     H�@     Ѓ@     h�@     ��@     ؃@     h�@     p�@     �@     ��@     Ё@     ��@     ��@      �@     �@     @@     (�@     Ѐ@     0�@     �@     @}@     @@     �}@     p}@     0~@     `~@     �|@     �@     P|@     `z@     �|@     �{@     @{@     �{@     pz@     �x@     0z@     �z@     0x@     @w@     x@     pw@     �v@     y@     �v@     �x@     �w@     �v@      v@     `t@     Pv@     �u@     �t@     �r@      u@     �s@     �s@     �s@      t@     �r@     �r@     s@     0q@     �q@     @s@     0s@     `p@     0p@     �n@     �q@     �p@      n@     �n@     �q@     @p@     �n@     `n@     @n@     �o@      n@      l@     p@      k@      m@      m@     �j@     �o@      m@     @l@     @l@     `j@     �j@     �i@     �k@     �k@     `h@     �h@      f@     �i@     �i@     �g@     �e@     @e@     �d@      f@     �c@      c@     �e@     �`@     �d@     �e@     �f@     �h@      f@     �e@      d@     �b@     @�@     c�@     �f@     @h@     �f@     `f@      h@     `f@     `i@     �j@     `k@     �h@      h@     �j@     �e@     �h@     `h@     �h@     @m@      g@     �h@     �i@      i@     �j@     �m@      j@     @n@      o@     �o@      m@     �l@     �k@     `n@     �n@     �o@      n@      p@     �p@     �m@     �o@     �q@     pp@     0p@     Pq@     0r@     @r@     pp@     �s@     �s@     �t@     0t@      q@     0s@     `s@     0t@     �s@     Pt@     0v@     u@     �t@     �v@     �u@      v@     pt@     �v@      v@     �w@     �w@     �z@     `w@     �x@     `x@     �v@     pw@     0{@      y@     �z@     �|@     0~@      {@     �|@      @     P{@      ~@     �}@      @      �@     ��@     @�@     `�@     p@     `@     ��@     �~@     �@     �@     P�@     ��@     X�@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     ��@     Ȅ@     ��@     ��@     �@     ��@     x�@     ��@     ��@     ��@     ��@     8�@     ��@     ��@      �@     �@     ��@     ��@     h�@     h�@      �@     �@      �@     h�@     X�@     ��@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ,�@     Џ@     H�@     4�@     $�@     D�@     �@     ̒@     �@     <�@     x�@     ȓ@     ��@     ��@     �@     P�@     P�@     Ȕ@     8�@     �@     �@     H�@     ��@     D�@     ̘@     ��@     К@     ��@     ��@     �@     ��@     D�@     ��@     ��@     \�@     Ġ@     Ƞ@     �@     Ƣ@     d�@     ʣ@     `�@     �@     n�@     �@     r�@     �@     Z�@     ک@     Ъ@     4�@     ��@     �@     b�@     ��@     H�@     �@     ��@     6�@     ĳ@     ��@     O�@     S�@     ɷ@     "�@     ո@     ɹ@     ��@     h�@     �@     ��@     ��@    ���@     ��@    �E�@    ���@     ��@    �o�@     &�@     ��@     ��@     ��@    ���@    ���@     ��@     )�@    ���@     �@    ���@    ���@     l�@     ��@     ��@    �F�@    @��@    @,�@    ���@    ���@    ��@     ��@     a�@    @k�@     J�@    ���@     b�@     ��@    ���@    �q�@    @&�@    �D�@    ��@    ���@    @��@    ���@    ��@    @��@    �P�@    �H�@    ���@     ��@     I�@    ���@    ���@    �9�@     ��@    ��@    ���@    ��@    @��@    ���@    ��@     ��@     ��@    ��@    ���@     L�@    @��@    ���@    �+�@    ���@    @��@    ���@    @��@    @@�@    ���@    @B�@    �r�@    @g�@    @y�@    ��@     ��@    �w�@    ��@     k�@     ��@    �$�@     \�@     O�@     ��@     ��@     ��@     �@     ��@      �@     ֣@        
�
predictions*�	   �)ȿ   �� @     ί@!  T�p'@)�H�I7@2��@�"�ɿ�QK|:ǿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�1��a˲���[��})�l a��ߊ4F����(��澢f�����uE���⾮��%ᾙѩ�-߾pz�w�7�>I��P=�>��Zr[v�>��[�?1��a˲?6�]��?����?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?ܔ�.�u�?��tM@�������:�              �?              �?      �?       @      @      (@      4@      8@      <@      H@     �J@      O@      P@     �R@      U@     �W@     �V@     �V@      W@     �T@     �U@     @V@     @V@     @Y@      P@     �U@      S@     �T@     �S@     �J@      K@     �M@      K@     �I@     �B@     �F@     �B@     �B@     �@@      <@      @@      9@      1@      2@      *@      1@      $@      *@      (@      "@      .@      $@      "@      @      @       @      @      @      @      @      @      @              @      �?       @              @       @      @       @      �?      �?      �?       @              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?              �?               @              �?       @      �?              @      �?      �?               @       @       @       @      �?      @      �?       @      @       @       @      @      @       @      @      @      @      @      @      "@      *@      "@       @      (@      .@      "@      (@      0@      1@      (@      5@      7@      9@      2@      7@      =@     �A@      @@     �B@      D@      C@     �J@     �H@      C@      E@      M@      G@      F@      A@      I@      A@      H@      C@      D@     �@@      D@     �@@     �A@      9@     �B@      <@      6@      8@      ,@      .@      6@      4@      2@      2@      *@      ,@      &@      "@       @      @      @       @      @      @      "@      @      @      @      @      @       @      @      �?      �?       @      �?              �?               @        k,o5r3      �i	p�0f���A'*�f

mean squared error�� =

	r-squared��8>
�L
states*�L	      �   ���@   �$[RA!]t��ײ��)��Zؖ*A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             k�@     ��@      �@     �@     ذ@     -�@     ,�@     ��@     ��@     B�@    ���@    ��@    0[�@    @�@    @k�@    �4�@    ��@    ���@    `J�@    ���@    ���@    ���@    `��@    `��@    @�@    @��@    ���@     ��@    �_�@    @T�@    �1�@    ��@    `.�@    �"�@     �@    @��@     k�@     q�@    `��@    @��@    �t�@    �[�@    @��@    `0�@    ��@    @��@    ���@     ��@    @��@    @J�@    @��@     �@    �n�@      �@    ���@    �U�@    ���@    ���@    ���@    ���@    �o�@     ��@    ���@     m�@    ���@     
�@    ���@     q�@    @��@    @�@    ��@    @��@     ��@    ���@     N�@    �B�@    ��@    ���@    ���@    ���@    ���@    �Z�@    �5�@     k�@     B�@     ��@     W�@    �s�@     2�@     l�@     ��@    ���@     ��@     I�@     ^�@     �@     �@     ��@     @�@     1�@     Է@     ��@     S�@     ܵ@     ^�@     ��@     �@     �@     ��@     �@     �@     0�@     ~�@     D�@     ʮ@     ��@     B�@     ��@     ��@     8�@     ��@     �@     �@      �@     ̥@     D�@     N�@     �@     p�@     F�@     ��@     ��@     H�@     P�@     �@     T�@     ��@     <�@     D�@     ��@     ��@     L�@     �@     H�@     ��@     �@      �@     <�@     ��@     X�@     ��@     ��@     ��@     ��@     ��@     (�@     @�@     ��@     x�@     ȏ@     ��@     P�@     �@     �@     Ќ@     ��@     ��@     �@      �@     ��@     ��@     ��@     ��@     H�@     `�@     ��@     ��@     P�@     8�@     ��@     ��@     ��@     X�@     ��@     �@     P�@     H�@     Ȅ@     ��@     Ѓ@     @�@     (�@     ��@     @�@     ؀@     ��@     h�@     P@     ��@      @     (�@     X�@     @     p@     �@     �}@     �~@     �}@     @}@      ~@     �{@     �{@     �{@     p�@      }@     0{@      |@     �|@     0{@      |@     �|@     �z@     �x@     �x@     `y@     x@     pv@      w@     �v@      x@     pu@      v@     @v@     �v@     �u@      u@     �u@     �t@     �q@     �u@     0r@      s@     Ps@      r@     `r@     `r@     �p@     �p@     Pq@     �r@     Pp@     �r@     �p@     �s@      r@      o@     Pq@     pq@     �q@      q@     �o@     �p@     Pq@      r@      n@      o@     �l@      l@     �i@     @k@     �j@     �k@     �j@      k@     �k@     �j@     �k@     �f@     �h@     `g@     `i@     �e@     �h@     @f@      f@     `f@     �f@     �h@     �e@     �e@      h@      h@      e@     �e@      e@     `f@     �e@      f@     �d@     @d@     �b@     �c@    ���@     ��@     �e@     �d@     �d@     �f@     `g@     �f@     `e@     �j@      k@     `f@     �h@     @j@      i@     @j@     �m@      k@     �l@     �k@     �m@     �l@     �m@     @k@      m@     �j@     @o@     @o@     �k@      m@     �m@     �o@     @o@     @p@     �n@     Pr@      p@     �p@     `q@      p@     `p@      s@     �p@     �r@      q@     Pq@     �q@      r@     �s@     �q@     @r@      s@      u@     �q@     Ps@     �s@     �s@     �t@      s@     �u@     `s@     0v@     �w@     �v@     @w@     �u@     @v@      y@     �x@     �x@      z@      z@      |@      y@      {@     pz@     P{@      {@     �{@     {@     @z@     p}@     0@     0{@     �}@     0@     ��@     0�@     �@     �}@     P�@     �~@     H�@     �@     �@     ��@     �@     �@     ��@     Ȃ@     ��@     H�@     ��@     Ђ@     ��@     �@     ��@     �@     ��@     ��@     �@      �@     ��@     �@     8�@     ��@     ��@     ��@     x�@     ��@     ��@     �@     ��@     ��@     ��@     �@     ��@     ��@     ��@      �@     ��@     ؋@     ��@     �@      �@     ��@     0�@     D�@     (�@     |�@     |�@     ��@     ��@     4�@     P�@     ��@     ��@     ��@     ��@      �@     �@      �@      �@     �@     T�@     0�@     @�@     8�@     <�@     h�@     ��@     H�@     X�@     |�@     ��@     8�@     ��@     t�@     �@     ��@     �@      �@     r�@     ��@     >�@     ��@     �@     ��@     D�@     ��@     >�@     �@     ʨ@     �@     n�@     ��@     ��@     �@     �@     Ư@     ��@     >�@     ױ@     q�@     8�@     ��@     ,�@     ��@     a�@     m�@     ˷@     ��@     ��@     5�@     ��@     9�@     ��@     �@     ��@     3�@     ��@     8�@     ��@    �5�@     L�@     ��@    ���@     ��@    �T�@     ��@     _�@    ��@     ��@     ��@     _�@    ���@    ���@     ��@     \�@    �2�@    �U�@     ��@     
�@    �R�@    @��@     ��@    @7�@    ���@    @L�@    @)�@    ���@     ��@    @��@    @\�@    ��@    ���@    ���@    �d�@     �@    @��@     }�@    �h�@    �@�@    ��@     d�@     Y�@    ���@    ���@    �q�@    ���@    �M�@    ���@    ���@     ��@    ���@    @��@    �<�@    @n�@    ���@     Y�@    �\�@    @�@    ���@    ���@    �4�@     ��@    �%�@    @a�@    ��@     �@    @��@     k�@     `�@    �e�@    ���@     T�@    �t�@    �8�@     ��@    @��@    ���@    p�@    �e�@    �d�@    ���@     ��@      �@     ��@     *�@     ��@     X�@     �@     8�@     �@        
�
predictions*�	   �t=��    �x@     ί@!  0#�)<�P�;F7@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.���T7����5�i}1���d�r�x?�x��6�]���1��a˲���[���FF�G �>�?�s���pz�w�7��})�l a��uE����>�f����>a�Ϭ(�>8K�ߝ�>I��P=�>��Zr[v�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?2g�G�A�?������?+�;$�?cI���?�P�1���?3?��|�?�E̟���?�Š)U	@u�rʭ�@�������:�              �?              �?      @      $@      6@      7@      <@      >@     �D@     �H@     �N@     �P@      M@     @T@     @T@     �S@     �V@     �X@      V@     @W@     �X@     �U@     �U@     �X@     �S@     �T@      R@     �U@     @Q@     @P@      H@     �O@     �P@      G@      I@     �F@      K@     �A@      D@      >@      8@      6@      =@      0@      8@      ,@      0@      *@      $@      &@      $@      &@       @      &@      @      @      @      @      @      @      @      @      @      @      @      @       @      @      @      �?      �?      �?       @      �?              �?              �?              �?              �?       @              �?              �?              �?              �?              �?              @              �?              �?              �?      �?      �?              @       @       @      �?      �?      @      @       @      @       @       @      @      @      @      @      @      @      $@      @      "@      ,@      &@      @      &@      (@      2@      6@      .@      2@      5@      8@      =@      :@      ?@      =@     �@@     �F@      9@      F@      F@     �A@      C@      J@      D@      D@     �G@     �G@     �D@     �F@      B@     �B@      6@      F@      >@      9@      =@      9@      4@      7@      *@      6@      3@      4@      (@      1@      @      (@      *@      &@      @      @      @      @      $@      @      @      @       @      @      �?      �?       @       @      �?      �?              �?              �?              �?      �?              �?              �?        ��ּ�3      �gw	Uj�f���A(*�g

mean squared errorA�,=

	r-squared�J�=
�L
states*�L	      �   ���@   �$[RA!�*�����)$yڽ��*A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ��@     "�@     ,�@     �@     p�@     f�@     J�@     C�@    ���@    ���@    ���@    ���@     ��@    ���@    ���@     ��@     ��@     ��@    @��@    @��@     ��@    @��@    ���@    `��@    ���@     ��@    �u�@     0�@     #�@     ��@     m�@    @�@     �@    @�@    �y�@    ��@    �<�@    `V�@     ^�@    �z�@     {�@    �<�@    �R�@    `R�@     �@     �@    @��@    �X�@    ���@     2�@    @��@    ��@    ���@    ��@    @��@     �@    �~�@    @S�@    �W�@    �{�@    @$�@    �|�@    @��@    �t�@    @��@    ��@     [�@    @��@    �"�@    �A�@    ���@    ���@    @n�@    ���@    ��@     ��@    ��@     ��@    �C�@    �y�@    �f�@    �"�@    ���@     z�@    ���@     �@    �h�@    ��@     ��@    �3�@    ���@     0�@     ־@     Ǿ@     6�@     T�@     Z�@     �@     ��@     b�@     �@     \�@     �@     ��@     ۵@     �@     8�@     O�@     ��@     ��@     T�@     ΰ@     ,�@     ̮@     �@     ��@     �@     ��@     ��@     �@     �@     �@     ��@     8�@     &�@     X�@     X�@     X�@     �@     :�@     Ρ@     ��@     j�@     ��@     �@     ��@     h�@     X�@     x�@     �@     �@     ��@     ,�@     0�@     ��@     �@     �@     ��@     ��@     ��@     ��@     L�@     \�@     �@     L�@     �@     ܑ@     ��@     ��@     �@     ��@     `�@     H�@     ��@     �@     ��@     ��@     h�@     ��@     ��@     �@     ��@     ��@     Ȋ@     p�@     ��@     8�@     ��@     ��@     ��@     �@     ��@     ��@     0�@     ��@     ��@     ��@     ��@     ��@     ��@     `�@     ��@     H�@     Ѓ@      �@     ��@     8�@     @�@     `�@     �~@     �@     `�@     ��@     @�@     P�@     P@      �@     }@     �}@     �@     �~@      }@     �~@      ~@     �{@     @{@     �|@     �{@      |@      {@     0}@     �y@     �x@     �y@     �y@     �x@     �x@      w@     0y@     0x@     pt@     `x@     pw@     �t@     �v@     x@      u@     �t@     pu@     `u@     �u@     �t@     Pq@     �r@      v@     �s@     0v@     �t@     �s@     ps@     pq@     �r@     �p@     `p@     `o@     �o@     �p@     �q@     �q@      o@     �p@      m@     @p@      p@     @p@      p@      q@     �o@     �p@     `m@     �l@      m@     �k@     �n@     `h@     �m@     �l@     `k@     �k@     �i@     �h@     �i@     @i@     �f@      g@      i@     �f@     �d@     `e@      k@     �h@     `d@      g@      k@     �j@     �h@     @g@     �j@     @c@     @d@     �d@      d@    �m�@     ��@     �h@      g@      h@     �c@      k@     `k@     �i@     @k@     @l@      h@     �k@     @k@     `l@      j@     @j@      i@     �k@     @j@     @m@      n@     �l@      m@     �o@     @o@      o@     �j@     0q@     `m@     �l@     �o@     �n@     �q@     pp@     �t@     Pp@     `r@     �q@     �q@     pt@     �u@     @t@     Pr@     �r@     `t@      r@     �s@     @s@     pu@     `t@     @v@     �u@     �u@     �t@     `v@     w@     v@     �v@     �v@     �v@     �x@      x@      {@     �w@     @{@     �z@      z@     0|@     �y@     �z@     �y@     �z@     �{@      z@     �z@     @~@     `{@     �|@     �|@     `{@     �}@     �}@     P}@     (�@     ��@     @     p�@     �@     �@     ��@     0�@     ��@     ؀@     �@     @�@     P�@     ��@     0�@     8�@     �@     ��@     ��@     �@     ��@     (�@     ��@      �@     (�@     ��@     ��@     ��@     ��@     ��@     ��@     `�@     ��@     P�@     �@     @�@     ��@     H�@     ��@     �@     ��@     ��@     ��@     h�@     �@     H�@     ��@     ��@     ��@     ��@     8�@     ��@     ��@     ��@     8�@     0�@     Џ@     X�@     ��@     ��@     ��@     �@     ��@     x�@     ��@     @�@     L�@     Д@     ��@     h�@     ��@     ��@     �@     H�@     ��@     �@     ��@     ��@     ��@     D�@     �@     ��@     d�@     ��@     Ğ@     ��@     �@     ��@     ~�@     ��@     ��@     ��@     �@     0�@     �@     2�@     @�@     ~�@     ��@     ԩ@     t�@     ��@     *�@     ެ@     D�@     �@     j�@     ��@     ��@     ��@     1�@     @�@     Ǵ@     s�@     4�@     ڶ@     ;�@     	�@     ��@     �@     =�@     ��@     W�@     ��@     /�@     ��@    ���@     *�@     ��@    �&�@     R�@     A�@    ���@     ��@    ��@    �B�@     R�@    ��@    ���@     ��@    �'�@    ���@    �e�@    ���@     %�@     �@    �%�@    ���@    ���@    ���@    @�@    @��@     �@    @��@     ��@    �1�@    @��@    ���@    @��@    �'�@     !�@     ��@    �y�@     ��@    @��@    ���@    @��@    @�@     ��@    ���@    ���@     ��@    @��@    ���@    @��@    �R�@    �u�@     $�@    ���@    �z�@    ���@    ��@    ���@    @��@     ��@    @7�@    @��@    �a�@    @��@    ���@    ���@    ���@    @��@     ��@    ���@    ���@    @��@     ��@    ���@     #�@     ��@    @Q�@    �E�@    @��@    ���@     ��@     I�@    ��@    ���@    ���@    ���@     ��@     /�@     %�@     E�@     ֯@     ��@     Ԙ@     ��@     �@        
�
predictions*�	   �Wn��   �Do�?     ί@!  ���D@)IƢ�)�2@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��5�i}1���d�r�x?�x��f�ʜ�7
�������FF�G �>�?�s���I��P=��pz�w�7��})�l a��ߊ4F��a�Ϭ(���(����_�T�l׾��>M|Kվ;�"�q�>['�?��>�ߊ4F��>})�l a�>pz�w�7�>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�P�1���?3?��|�?yL�����?S�Fi��?�������:�               @      @      (@      ,@      4@      .@      9@     �C@     �D@      E@     �D@     �G@      K@      O@     �G@     �H@      G@      J@      L@     �G@      Q@      H@      J@      F@     �D@      F@     �E@     �A@      I@     �B@      B@     �@@      7@      :@      9@      6@      3@      8@      ;@      0@      6@      0@      4@      2@      ,@      "@      *@      &@      &@      *@      @      $@      @      @       @       @      @      @      �?      @      @      @       @      �?       @      �?      @       @      �?               @              �?              �?      �?              �?              �?              �?      �?      �?              �?              �?              �?              �?      �?              �?               @      �?              @              �?               @              �?              �?               @              �?      �?      �?      @      @      @      @      @      @      @      $@       @       @      @      @      @      @      @      &@      $@      ,@      *@      ,@      1@      4@      <@      3@      >@     �G@     �A@     �E@     �D@     �F@      I@      L@     �N@     �Q@     �R@     �S@     �T@     �W@     �Z@     �U@     �S@     �R@     �U@     �U@      S@     �J@     �M@     �J@     �J@     �H@      H@      E@     �A@      H@      >@     �C@      <@      ;@      ;@      6@      6@      $@      *@      &@      "@      "@      $@       @      @      $@      @       @      @      @      @      @      �?      �?      @      �?      �?       @      �?              �?      �?              �?               @        �� r3      �i	V�2g���A)*�f

mean squared errore %=

	r-squared�~">
�L
states*�L	      �   @��@   �$[RA!�th���)�	��r ,A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ҳ@     �@     ��@     *�@     �@     ��@     7�@    ��@    ���@    ��@    �(�@     ��@    ���@    �X�@    ���@    `�@    �e�@     ��@     ��@    ���@     |�@    ���@    `O�@    �H�@    ���@    �I�@    @u�@    �5�@    ���@    �a�@     ��@    �?�@    �y�@     ��@     ��@    �5�@    ���@    `�@    @��@    ��@    ��@    ���@    `�@    ���@     {�@    @x�@    ��@    @��@    @*�@    ���@    @v�@    @��@    �c�@     ��@    @ �@     ��@    @)�@    @q�@    �+�@    ���@     ��@    �?�@    @�@    �{�@    ���@    @1�@    �=�@    @��@    ��@    �d�@     �@    ��@    ���@     ?�@    ���@     ��@    �&�@     ��@    �d�@    ���@    ���@    ���@    ���@     u�@     �@    �q�@    ���@    �'�@    ���@     ��@    �J�@    ���@    ��@     D�@     z�@     ��@     ��@     '�@     �@     3�@     ��@     Ϸ@     ݶ@     �@     N�@     �@     /�@     ߳@     	�@     ��@     ��@     D�@     ��@     X�@     ��@     ��@     �@     Ԭ@     �@     ��@     x�@     �@     �@     @�@     �@     2�@     ��@     ��@     v�@     ��@     �@     �@     ��@     ��@     ��@     L�@     �@     �@     ��@     <�@     ܙ@     ��@     ܙ@     ܘ@     �@     ��@     @�@     ��@     �@     $�@     ,�@     X�@     ��@     $�@     X�@     ��@     ��@     D�@     Ȑ@     ��@     ��@     ��@     ��@     p�@     (�@     X�@     P�@     ȍ@     @�@     ��@     �@     H�@      �@     �@     p�@     ��@      �@     Ȉ@     p�@     ؇@      �@     X�@     �@     ��@      �@     ��@     x�@     `�@     ȃ@     Є@     h�@     �@     Ѓ@     h�@     ؂@     ��@     @�@     �@     �@      �@     h�@     ��@     ��@     ��@     p�@     �@     @@     p�@     ��@     �~@     @@     �}@      �@     �~@     �}@     �@     �{@     �~@      {@     @{@     �z@     �z@      |@      |@     �z@     �z@     �x@     �w@     Pz@     �w@      z@     �w@     �w@      w@     �x@     �w@     0u@     Pu@     �v@     t@     u@     �v@     Pv@     u@     �t@      t@     �t@     s@     `s@     �r@     �q@     �r@     �r@     `s@     �q@      r@      r@     `r@     �p@     0r@     @m@     �q@     �p@     @m@     �p@     �l@     �n@     �p@     �l@     �p@     @m@     �n@      m@     @i@     `j@     @m@     �n@     @n@      l@     `k@     `j@     �i@     �j@     �k@     @h@      l@     �h@     �h@     �e@     �f@      j@     `j@      i@     `h@     �e@     �e@     �c@     �f@      j@     �e@     �e@     �f@     ��@    ��@     �h@      j@     �k@     �g@     `j@      k@     `i@      h@      j@      l@      k@     `m@     @i@     �k@     �l@     �n@     �j@     @o@     @l@     `o@     pq@     �p@     �o@     �n@     `p@     @p@      p@     �p@     `r@     �n@     �q@     r@     �p@     �q@     �r@     �q@     s@     0q@     �s@      t@     ps@     �u@     pt@     0u@     @u@     �s@     �s@      v@     �s@     0u@     �t@      v@     Pt@     w@     �w@     �x@     �x@     `x@     �x@     �y@      x@     P|@     �z@     �y@     0~@     �x@     x@     �x@     �y@     py@     �{@     �y@     �~@      ~@     {@      }@     �{@     �|@     0~@      ~@     �@      }@     0�@     �@     `�@     `�@     ��@     ��@     ��@     X�@     ��@     ��@     ��@     (�@     ��@     �@     h�@     ��@     ȃ@     ��@     ؄@     ��@     ��@     ��@     P�@     ��@     @�@     8�@     @�@     ��@      �@     0�@     0�@     8�@     ��@     ��@     ؈@     ȇ@     �@     ��@     ��@     ��@     ��@     ��@      �@     P�@     �@     �@     �@     �@     T�@     @�@     �@     �@     L�@     ��@     �@     @�@     L�@     <�@     ��@     H�@     �@     �@     d�@     T�@     ��@     ��@     ��@     �@     ��@     $�@     L�@     ��@     ��@     `�@     ��@     x�@     ��@     p�@     P�@     ��@     ,�@     4�@     �@     �@     ܠ@     ��@     ��@     `�@     ��@     "�@     ��@     �@     ��@     R�@     V�@     ��@     ާ@     ب@     �@     ��@     ȫ@     ��@     ��@     �@     p�@     ��@     �@     ��@     ձ@     ��@     ��@     ��@     �@     ��@     u�@     E�@     ��@     ׸@     ��@     ��@     ��@     x�@     �@     ��@     ��@     �@     ��@     G�@     o�@    �Q�@     ��@    �"�@    �
�@    ���@    ��@     ��@    �/�@    ���@     ��@    �6�@     L�@     �@    ���@    ���@    ��@     ,�@     �@    ���@    ���@     x�@    ���@     ��@    � �@    ���@    @�@    @ �@    ���@    @��@    �1�@    ���@    ���@    ���@      �@    ���@     F�@    ��@    @!�@    @��@    ���@     #�@     B�@    @K�@     o�@    �X�@    @��@     ��@    ���@    ���@    ���@    @��@    ��@    �M�@    ���@    @��@    ���@    ���@    � �@     ��@    ���@    @<�@    ��@     7�@    @1�@    @��@    �O�@     ��@    �l�@    @`�@    @��@    �S�@    @��@    @~�@     ��@    @�@     ��@    `�@    ���@    �v�@    ��@    ��@    �1�@    ��@     Q�@     ��@     ��@     f�@     ��@     H�@     F�@     �@        
�
predictions*�	   �	[��   @�J@     ί@!  f�=:@)D���9=A@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r������6�]�����(��澢f������>M|Kվ��~]�[Ӿjqs&\�ѾX$�z��
�}����})�l a�>pz�w�7�>��[�?1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?�P�1���?3?��|�?�DK��@{2�.��@�������:�              @      �?      @      @      (@      7@      4@     �B@     �H@      A@      J@     �L@     �M@     �O@      M@     �T@     �P@     �Q@     �V@     @R@     @P@     �R@     �R@     @P@     �P@      I@     �M@      O@      L@     �E@      H@      G@      J@     �D@      B@     �@@     �C@      @@     �A@      ;@      4@      3@      :@      1@      .@      0@      (@      "@      @      0@      *@      @      @      (@      @       @      @      @      @      @      �?      @      �?      @              @       @      @      �?      �?      �?              @              �?      �?      @      @      �?      �?              �?              �?              �?      �?              �?              �?              �?              �?      �?      �?      �?      �?       @              �?              �?       @              �?      �?      �?      �?      @              @      @       @      @       @      @      @       @      @      @       @      @      @      @      @      0@      (@      &@      *@      1@      2@      6@      1@      $@      3@      4@      6@      :@      ;@      2@     �G@      J@      H@     �A@      E@      J@      H@      M@     �L@     @P@     �M@     �S@      O@     @R@      K@     �O@      M@     �I@      F@      F@      F@      F@      D@      B@      G@      >@      7@      7@      6@      4@      .@      *@      *@      ,@      *@      "@      @      @      @      $@      "@      "@      $@       @      @      @      @      @      @       @              �?       @      �?      @              @              �?        ��$��2      ���	�a�g���A**�e

mean squared error�\=

	r-squared��?>
�L
states*�L	      �      @   �$[RA!Α֒����)���D.A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ϳ@     ԧ@     ��@     �@     ʸ@     7�@    ���@    ���@    �B�@     -�@    �n�@    �n�@    p��@     ��@    �N�@    ���@     ��@    `a�@    ���@    ���@    @*�@    ��@    @1�@    ���@    @��@    @��@    @�@     ��@     }�@    ���@    ���@    ���@    �u�@    @��@     �@    @`�@    ���@    ���@    ���@    ���@    ���@    @��@     ��@    ���@    @��@     ^�@     ��@    @��@    �M�@     ��@    ���@     3�@    @��@    @?�@     ��@     ��@     �@    ���@     E�@    @��@    @s�@    ���@    ���@     '�@    �r�@     ��@    ���@    ���@    �_�@    ���@    @<�@    @��@     Z�@    �e�@    �J�@     ��@    ���@     ��@     ��@    ���@     f�@     ��@     ��@    ���@     a�@    ���@     ��@    �6�@     ��@     �@      �@     ��@     �@    ���@    �/�@     >�@     Q�@     V�@     �@     �@     ��@     p�@     �@     ��@     l�@     ڷ@     0�@     U�@     ��@     Ĵ@     `�@     �@     C�@     ��@     �@     $�@     ǰ@     ��@     �@     ��@     ��@     ��@     �@     ��@     ��@     x�@     ��@     *�@     b�@     ��@     Φ@     ܥ@     �@     ��@     ��@     Ģ@     ~�@     ��@     ̡@     �@     0�@     ��@     ̞@     ��@     �@     d�@     Ț@     �@     ��@     D�@     $�@     ��@     `�@     @�@     x�@     4�@     ܓ@      �@     ,�@     l�@     ��@     �@     �@     �@     ��@     4�@     Ȏ@     �@     p�@     x�@     ��@     �@     X�@     ��@     `�@     H�@     ��@     ؈@     `�@     ��@     �@     �@     H�@     ��@     ��@     X�@     x�@     x�@     Ȇ@      �@     X�@     @�@     ��@      �@     ��@     ��@     (�@     ��@     ��@     ��@     �@      �@     0�@     �~@     H�@     h�@     ��@     �@     `}@     ��@     (�@     �{@     �|@     �~@     p}@     �~@      }@     �z@     �{@     Pz@     `{@     �|@     0z@     �{@     �z@      z@     �|@     �v@     @w@     Px@     �w@     �v@     �v@     @x@     �v@     �t@      v@     �u@     �t@     �s@      w@      u@     Ps@     Pt@     �s@      t@     �u@     0s@     pt@     �s@     pt@      s@     �n@     �q@     �p@     `s@     �q@     `q@     `s@     �o@     `o@     �p@     �p@     �q@      n@     �l@     �m@     @n@      o@     �i@     �j@      m@      o@      n@     @k@     @k@     @h@     �i@     �l@      i@     �l@      h@      k@      j@     �i@     `i@     �f@     �f@     �e@     @e@     �e@     �d@     �d@     �e@      e@     �e@     @g@     �j@     �i@     `d@     �d@      c@     ��@     ��@     �j@     @j@      m@     �f@     �h@     `j@     �g@     �g@     �i@      i@     �m@     �k@     �n@     �m@     �m@     �n@     �p@     @k@     �n@     `q@     �o@     �p@     �o@      p@     �o@     �n@     @p@     �o@     �p@      p@     �o@     r@     �q@     �t@     �q@     `s@     �r@     0r@     @t@     `t@     �s@     �s@     �v@     ps@     �t@     �w@      t@     `u@     0t@     @u@     u@     0v@      u@     �v@     �w@     `w@     �x@     �x@     �{@     �x@     y@     �x@     �w@     z@     �{@     �{@     `|@      {@     �z@     �|@     `|@     �z@     p@     �|@     �|@     P~@     �@     �|@     8�@     ��@     ��@     ��@     p@     0�@     X�@     �@     Ѐ@     ��@     ��@     ��@     @�@     �@     �@     (�@     Ѓ@     �@     �@     �@     �@     ��@     ��@     ��@      �@     (�@     P�@     ��@     ȅ@     `�@     ��@     ؆@     ��@     H�@     P�@     �@     ��@     x�@     ��@     x�@     X�@     ��@     H�@     ��@     ��@     `�@     H�@     `�@     ؎@     Ў@     ��@     ،@     X�@     `�@     �@     ��@     �@     |�@     ��@     ��@     ��@     ��@     ��@     ��@     Д@     H�@      �@     <�@     8�@     ��@     |�@     �@     �@     ��@     �@     �@     d�@     ��@     ��@     t�@     ԟ@     ��@     ��@     Ƣ@     4�@     V�@     b�@     |�@     ��@     T�@     Ħ@     :�@     ��@     ��@     �@     
�@     ��@     ��@     �@     ��@     ��@     �@     �@     ��@     3�@     ȱ@     ��@     ��@     a�@     ��@     ��@     `�@     ��@     ��@     ��@     o�@     ��@     ߸@     ��@     V�@     ��@     ��@     O�@     �@     ��@     ��@     W�@    �J�@     h�@    ���@     k�@     ��@     D�@    ���@     ��@    ���@    �a�@     ,�@    ���@    ���@    �c�@    ���@     ��@     {�@    �m�@     K�@    �^�@     `�@     ��@     
�@    ���@     ��@     f�@     7�@    @��@    @b�@    �l�@    @��@    ���@    �5�@    ���@    ���@     E�@     ��@    �X�@    @��@    �T�@    ���@    �c�@    ���@     ��@    �1�@    ���@    �@�@    ���@     ��@    @��@    �W�@    @;�@    ���@     ��@     ��@    @��@    ���@    @��@    ���@    �#�@    ���@     �@     ��@     ��@    �q�@    @��@    ���@    @��@    ���@    �-�@    @�@     Y�@    �\�@    ���@    ���@    ���@    @N�@    �t�@    � �@    �i�@    @]�@    �`�@    �_�@     O�@     �@    �L�@     ��@     �@    �(�@    ���@     Y�@     ,�@     ��@     >�@     �@     ��@     ʪ@        
�
predictions*�	   ����   `I�@     ί@!  �b�5�)r%e5@@2��?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[��O�ʗ��>>�?�s��>�FF�G ?��[�?��d�r?�5�i}1?�T7��?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?!��v�@زv�5f@�������:�              �?      @      @      (@      8@      A@     �D@      F@     �J@     @P@     �Q@     �R@     @V@      U@     �V@     �R@     @T@      U@     �T@     �U@     @T@     �V@     @T@     �T@     �T@     �O@      O@     �L@      K@     @P@      J@     �H@      J@     �B@     �E@      G@      B@      B@     �@@      ?@      5@      3@      9@      *@      1@      4@      *@      *@      0@      @      @      ,@      @      "@      &@       @      @       @      @       @      �?      @      @       @      @       @      �?      @      �?      @              �?       @               @              �?              �?      �?      �?      �?              �?              �?              �?              �?      �?              �?       @       @              @       @              @      �?      �?      @      @       @      @      @       @      @      @       @      @      @      @      @      "@      *@      @      @      $@      *@      (@      6@      3@      4@      2@      =@      :@      D@      @@      ;@     �C@     �A@      H@      J@      I@      K@     �K@     �M@      H@     �F@      I@      F@     �H@      I@     �D@     �B@      D@     �D@      8@      B@      5@      7@      ;@      3@      5@      5@      2@      4@      2@      ,@      ,@      ,@      ,@      "@      @      @      @      @      @       @      @       @      �?      @      �?       @      �?              �?               @      �?      �?      �?      �?      �?              �?        qy��2      �	�x h���A+*�e

mean squared error�w'=

	r-squared�>
�L
states*�L	      �      @   �$[RA!y��y���)�}1��-A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             5�@     ��@     X�@     �@     �@    �
�@    �j�@    ���@    ���@     ��@    ��@     ��@    ���@     ��@    @��@     ��@     ��@    ���@     ��@    �W�@    ���@    @��@    ��@    @E�@    @,�@    @��@    @��@    �F�@    �&�@    ���@    @��@     B�@    �p�@    ���@    @��@    ���@    ��@    �l�@    @:�@    @��@    ���@    @�@     ��@    @��@     S�@    ���@    ���@     ��@    @~�@    ��@    @��@    @�@     ��@    �q�@    @�@    ���@    �2�@    @��@    �t�@    ���@     6�@    ���@    ��@    @��@    ���@    @e�@    ���@    ���@    �!�@    ���@    ��@    �R�@    @.�@    �T�@     
�@     ��@    ���@     ��@    ��@    ���@    ���@    ��@     Q�@    �~�@     ��@    �W�@    ��@     ��@     *�@    �]�@    �8�@     >�@    ���@     -�@     ��@     �@     *�@     м@     ٻ@     ��@     �@     Y�@     ݸ@     �@     ޶@     ȶ@     ��@     ��@     ȴ@     z�@     ��@     �@      �@     	�@     j�@     n�@     |�@     �@     Ԯ@     ҭ@     ��@     ��@     l�@     $�@     ��@     �@     ��@     �@     �@     ��@     D�@     �@     V�@     t�@     *�@     *�@     ��@     ��@      �@     ��@     ��@     ��@     �@     ��@     ��@     ��@      �@     �@     �@     �@     �@     �@     D�@     ��@     @�@     ܓ@      �@     ��@     �@     4�@     �@     ��@     ��@     �@     (�@     4�@     ��@     8�@     `�@     ��@     Џ@     h�@     @�@     �@     ��@     ؉@     �@     ��@     ��@     @�@     @�@     ��@     H�@     ��@     Ї@     ��@     ��@     ��@     X�@     ��@     `�@     �@     (�@     Є@     �@     0�@     `�@     ȁ@     ��@     �@     P�@     @�@     ��@     ��@     ��@     Ȁ@     p�@     �}@     �~@     `}@     h�@     �~@     ��@     �~@     P�@     �~@     �}@     |@     �|@     P~@     �{@     @|@     p{@     @|@      |@     `z@     �{@     �z@     �x@     �z@      y@     `x@     @x@     �y@     �y@     �x@     �z@     0z@      x@     `u@     u@     �w@     �t@     Pv@      t@     �t@     @v@     0t@     �w@     �u@     ps@     �q@     �q@     �s@     �q@     �p@     �p@     �q@     �r@     �o@     �q@      q@     �q@     0q@     �p@     �q@      q@     �n@     @n@     `m@     �n@     @n@     �m@     �m@     �n@     �n@      m@      k@     �k@     �h@     �k@     �k@      k@     �i@     �h@      m@     �k@     `e@     �i@     �j@     �j@     @k@     �g@     �h@     `i@      g@      g@     @d@      g@     @e@      f@     �d@     ��@    �m�@      j@     0p@      i@      k@      l@     �i@      i@     @j@     �l@      k@     @k@     `m@      l@     @o@      o@     �p@     Pp@      o@     `n@     `n@     0q@     �l@     `p@     `o@      p@     pp@     �o@     p@     pr@     �q@      r@     �r@     �r@     �u@     �t@     �s@     Pt@     �u@      t@     0u@     �t@     pt@     @t@     �v@     �s@     v@     �v@     Pv@     �t@      v@     `v@     0v@     �v@     �w@     �y@     �w@     `w@     w@     �w@      y@     �x@     �z@     �w@     0|@     `{@     �z@     �w@     �{@     {@     �|@      ~@     �~@     0@     x�@     }@     �~@     ��@      @     ��@     P�@     �@     �@     @�@     ��@     �@     p�@     �@     P�@     p�@     H�@     �@     `�@     h�@     x�@     `�@     �@     @�@      �@     X�@     �@     ��@     ��@     ��@     8�@     ��@     (�@      �@     p�@     ��@     ��@     ��@     ��@     ��@      �@     ��@     0�@     h�@     ��@     ��@     ��@     ��@     �@     �@     h�@     ��@     ��@     �@     P�@     ��@     (�@     t�@     �@     ��@     ��@     Ȑ@     ��@     $�@     ��@     ��@     `�@     ��@     �@     p�@     ��@     ��@     x�@     �@     ��@     ��@     З@     t�@     |�@     8�@     �@     �@     d�@     ��@     ț@     ��@     ��@     �@     �@     ��@     ��@     �@     �@     ,�@     ��@     ^�@     ��@     ��@     \�@     f�@     ��@     b�@     ��@     ��@     �@     �@     ��@     Ƭ@     ��@      �@     p�@     5�@     ��@     ��@     ��@     ]�@     ��@     �@     *�@     Y�@     a�@     A�@     i�@     �@     2�@     '�@     ¹@     ��@     T�@     w�@     2�@     �@     �@     �@    ��@    ���@     7�@    �!�@    �_�@     ��@     -�@     	�@    �\�@    �4�@    ���@     }�@     �@    ���@    �	�@    ���@    ���@     7�@    ��@    �	�@     ��@     ��@    ��@     ��@    �F�@     ��@     ��@     H�@    @��@    @��@    @�@    @��@    ���@     .�@    @��@    ���@    �F�@     ��@    @2�@    @��@    �.�@     ��@    @��@     �@     ��@    ���@    ���@    �J�@    ���@     {�@     ��@    �R�@    ���@    @��@    @��@    ���@    �<�@    �1�@     �@    ��@    ���@    ���@    ���@     E�@    ���@    @��@     ��@    �L�@    ���@    @�@    �Z�@     b�@    �l�@     ��@    ���@    �E�@     I�@    @��@    @�@    `��@    �*�@    �X�@     b�@     ��@    ��@     |�@     `�@     �@     �@     ��@     j�@     T�@     <�@     �@     <�@        
�
predictions*�	   �׉��   �9E@     ί@!  w�RW@)u��|ZK@2�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��S�F !�ji6�9���.����ڋ��vV�R9��T7����iD*L�پ�_�T�l׾�u��gr�>�MZ��K�>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?����?f�ʜ�7
?��ڋ?�.�?ji6�9�?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?S�Fi��?ܔ�.�u�?زv�5f@��h:np@�������:�              �?      �?              @      @      @      5@      7@     �@@      A@     �D@      B@     �C@      K@      F@     �N@      K@      F@      @@      L@     �G@      L@     �C@      A@      @@      A@      ?@      :@     �B@      :@      2@      9@      5@      *@      .@      $@      6@      4@      &@      $@      (@      @       @      $@      $@      @       @      @      @      @      @      @      @      @      @       @       @      @      @      @      �?      �?      �?       @      �?      �?       @      �?              @              �?              �?              �?              �?              �?      �?      �?              �?      �?               @      �?               @      @      @       @              @               @      @       @      @      @      @      @      @      @       @      @      @       @      @       @      @       @       @       @      *@      $@      8@      1@      6@      2@      7@     �D@      <@      @@     �F@     �F@     �E@     �I@     @S@     �S@     @T@      U@     @T@     �Z@     �[@     �Z@      _@     @Z@     �\@     �Y@      V@      W@      X@      S@      T@      S@      M@      I@      L@     �C@      D@      F@      7@     �D@      7@      <@      5@      3@     �A@      1@      6@      1@      "@      ,@      &@      @       @      @      @      @      �?      @              �?      @      @      @       @      �?               @      �?              �?              �?        rzj+�2      �m�	?��h���A,*�e

mean squared errorL�0=

	r-squared���=
�L
states*�L	      �      @   �$[RA!�x�d���)�c@��/A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     ��@     ��@     ư@     Ľ@    ���@    ��@     ��@    ���@    ���@    ��@    ���@    ���@     /�@     ��@     Z�@    ��@    ���@     ��@     �@    @^�@    ��@     �@    @��@    �>�@     3�@    @T�@    @��@    ���@     ��@    ���@    ���@    ���@     �@    �[�@    ���@    ���@    @y�@    @��@    @t�@     i�@    @��@    ���@    ���@     a�@    �f�@     �@     ��@     I�@    ���@    �K�@    ��@    ���@    ���@    ���@     ��@     ��@    @��@    �#�@    ���@    ���@     ��@    ���@     �@    @��@     Q�@    ��@    �|�@    ���@    ���@    ���@    �]�@     ��@    @��@    @�@     K�@    ���@     T�@    �P�@    �8�@     W�@    �V�@     {�@    ���@     #�@    ���@     l�@    �<�@     ��@     7�@    �e�@     �@    �C�@    �0�@     ��@     #�@     ޿@     ��@     ��@     ��@     �@     ^�@     �@     L�@     �@     �@     ��@     ܷ@     T�@     ��@     ��@     ��@     ڴ@     Ѵ@     e�@     ۳@     {�@     Ȳ@     ��@     T�@     >�@     y�@     Ȱ@     ڰ@     Q�@     )�@     $�@     ��@     �@     �@     l�@     �@     �@     2�@     Z�@     Ҩ@     ��@     �@     H�@     ҥ@     v�@     8�@     ��@     t�@     ��@     p�@     V�@     ��@     l�@     �@     d�@     ��@     ��@     ��@     x�@     �@      �@     ��@     �@     `�@     �@     |�@     p�@     �@     d�@     ؔ@      �@     ��@     l�@     p�@     0�@     ȑ@     ��@     ��@     h�@     @�@     �@     ��@     �@     Ѝ@     X�@     ��@     ��@     ��@     ��@     ؋@     ��@     ��@     ��@     ��@     @�@     ��@     8�@     �@     ��@     0�@     ��@     ��@     (�@     ��@     @�@     ��@     ��@     ��@     X�@     h�@     h�@      �@     �@     H�@     ��@     �~@     �@      @      {@     0~@     �{@     �~@     0}@     @{@     �{@     `z@     �{@     {@     y@     y@     �{@     {@     0y@      z@     �x@     Py@     �v@     �w@     �u@     0w@     �u@      v@     �v@     �v@     pt@     �u@     @w@     �u@     �t@     �u@     �t@     �v@     �r@     pt@     �t@      u@     0s@     �s@     0p@     �s@     �p@     Pr@     �q@     �p@     0s@     p@      p@     �p@     �p@     Pq@     �p@     �n@     @o@      n@     �n@     �n@      k@     @o@     �m@     `n@      m@     �j@     @m@     �k@     �i@     @i@     �k@     @i@     @k@     �g@     �j@      k@     �k@     @i@     @h@      f@     @g@     �h@     `j@      h@      i@      d@     `f@     �f@      d@     �c@     3�@    �;�@     `j@     �g@     @i@     `h@      k@     �m@     �l@     @g@     �j@     �i@     �i@     �m@      m@     �m@      j@     �p@     `n@     �p@     �m@     �o@     �q@      r@      p@     �m@     Pp@     `n@      o@     �p@      r@     @s@      s@     �r@     �r@      r@     �s@     `s@     �q@     @u@     `q@     �u@     �s@     `t@     �s@     �u@     pv@     Pu@     ps@     �t@     �t@     �w@     �v@     w@     u@     �w@      v@     �w@     �v@     w@      w@     @w@     �x@     �x@     0y@     py@      z@     �@     �}@     @}@     �|@     P}@      }@     P{@     �|@     �|@     �|@     P|@     `{@     P}@      ~@     �~@     X�@     �@     �@     �@     X�@     �@     x�@     ��@     ȁ@     ��@     �@     ��@     ��@     ��@     (�@     (�@     P�@     ��@     H�@     �@     ��@     ��@     0�@     ��@     ��@     H�@     X�@     ؈@     ��@     `�@     �@     x�@     �@     �@     p�@     ��@     ��@     ��@     x�@     x�@      �@     h�@     ��@     4�@     ,�@     D�@     �@     X�@     ��@     �@     Г@     L�@     ��@     0�@     x�@     ,�@     P�@     \�@     T�@     ��@     h�@     P�@     ��@     �@     <�@     X�@     ��@     �@     ��@     �@     �@     @�@     �@     T�@     ��@     ~�@     �@     ��@     �@     ��@     T�@     ��@     �@     �@     ��@     ��@     �@     ��@     �@     �@     �@     
�@     �@     m�@     �@     ڰ@     F�@     -�@     �@     l�@     -�@     �@     t�@     ��@     ��@     �@     ��@     �@     3�@     Ʒ@     5�@     ��@     }�@     m�@     `�@     �@     �@     3�@     ��@     ӽ@     ��@     ��@     p�@    ��@    �K�@    ���@     ��@     ;�@     ��@     ?�@     ��@    ���@    �Q�@     ��@    ��@     ��@    �\�@    ���@     d�@    ���@     j�@      �@     �@    �E�@    ���@     ��@    ���@     ��@     ��@    �w�@    ���@    �j�@     #�@     k�@    @2�@    @��@    @�@    �"�@    �m�@    ���@    ��@    ���@    ���@     �@     6�@     ��@    @��@     ��@    ��@    ��@    ���@    ���@    ���@    ��@    ���@    ���@    �g�@    @�@    @=�@    ��@    @E�@    ��@    ���@     ��@     K�@    ���@    @G�@    �M�@    ��@     q�@    ���@     n�@     ��@    @��@    ���@    �3�@     K�@    @K�@    ���@    @��@    @��@     b�@    @��@    ���@    ��@     �@     V�@    `�@    ���@    ���@     �@     "�@    ���@     k�@     p�@     y�@     ��@     �@     @�@     ��@     D�@     J�@        
�
predictions*�	   ��^��   ���	@     ί@!  �[3zM�)e�&��46@2�Ӗ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��vV�R9��T7���x?�x��>h�'��f�ʜ�7
������pz�w�7��})�l a��ߊ4F��jqs&\�ѾK+�E��Ͼ['�?�;�ѩ�-�>���%�>�uE����>�f����>6�]��?����?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?W�i�b�?��Z%��?+Se*8�?uo�p�?2g�G�A�?������?cI���?�P�1���?yL�����?S�Fi��?�Š)U	@u�rʭ�@�������:�              @              @      (@      0@      D@     �E@     @R@     �K@     @U@     �Z@     �W@     �\@      b@     @^@     �`@     @^@      `@     �^@     `b@     �c@     �]@     �`@     �]@      Z@     �W@     �Z@      X@     �W@     @S@     �Y@      O@     �H@     �D@     �A@     �K@      D@     �@@      4@      @@      1@      8@      8@      6@      0@      2@      1@      ,@      "@      @       @      @      @      @      $@      @      @      �?      @      �?      �?      @      �?      @              @      @      �?      @      @       @              �?              �?      �?       @              �?      �?              �?      �?              �?              �?              �?              �?              �?              �?      �?              �?       @      @              �?              �?      @       @       @      @      @      @      @      �?      @      @      @       @      �?      @      $@      �?      "@       @      .@      &@      @      3@      "@      .@      .@      5@      0@      4@      0@      3@      7@      1@      4@      5@      1@      0@      7@      0@      3@      2@      (@      (@      .@      1@      0@      0@      &@       @      @       @      @       @      @       @       @      @      @      @      �?      @       @      �?       @      @      @      @       @      �?      �?              @      �?      @      �?              �?              @      �?      �?              �?              �?              �?        ]���3      �gw	n�i���A-*�g

mean squared error��2=

	r-squared�1�=
�L
states*�L	      �      @   �$[RA!���`0s��)m��V�0A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             *�@     Ч@     ��@     ��@     �@    ��@     S�@     ��@    �\�@     �@    ���@     ��@    ���@    @��@    `��@    �F�@     ��@     ��@    ���@    �T�@    ���@    ���@    ���@    �S�@    ���@     ��@    @��@     V�@    @��@    �+�@    @��@    ��@    �W�@     e�@    ���@    @�@     ��@    ���@    ��@    ���@    @C�@    ���@    ���@    @��@     ��@    �t�@    �U�@     1�@    ���@     ��@    ��@    ���@     ^�@     ��@     ��@    @M�@     ��@    ���@     ��@    @G�@    ���@     U�@     ��@    ���@    ��@     ��@    @7�@    ���@    @i�@    ���@     ��@     ��@    �i�@    @��@    �c�@     ��@     ��@     ��@    ���@    ���@    ���@    ���@    �?�@    ��@     g�@     e�@    ���@    �N�@     .�@    ���@     ��@    �'�@     ��@    �A�@    ���@     s�@    ��@     �@     ��@     ��@     ��@     ��@     r�@     ,�@     ��@     /�@     ��@     7�@     �@     ��@     �@     �@     ��@     �@     A�@     ׳@     ��@     ��@     �@     b�@     �@     �@     -�@     Ű@     �@     p�@     (�@     ֭@     ��@     D�@     $�@     ��@     $�@     �@     ��@     �@     �@     ��@     �@     <�@     ȥ@     ��@     Ƥ@     ��@     ܢ@     2�@     (�@     �@     ��@     ��@     ��@     D�@     |�@     0�@     p�@     L�@     ��@     x�@     ��@     ��@     ��@     x�@     ��@     ȕ@     ĕ@     D�@     <�@     ,�@     0�@     `�@     �@     D�@     ��@     ��@     ��@     �@     ��@     x�@     ��@      �@     Ћ@     ؍@     ؋@     Ȋ@     ��@     ��@     ��@     ��@     ��@     ؈@     �@     @�@      �@     h�@     0�@     x�@     x�@      �@     h�@     ��@     h�@     ؂@     ��@     p�@     Ȃ@      �@     Ё@     ��@     �@     ��@     ��@     P�@     ��@     @     ��@     �~@     P�@      }@     p}@      {@     Py@     �|@     `{@     �}@     |@     �y@     `}@     `y@      {@     �y@     @x@     �y@     �x@     x@     �x@     px@      v@     �u@      v@     �u@     �t@     v@     �s@     �v@     �r@     �u@     @v@     �u@     0t@     v@     �t@     �u@     �s@     �s@     �r@     Pr@     �q@     @p@     �q@     q@     q@     q@      r@     �r@     �q@     �q@     0q@     `r@     �q@      q@     �o@     �p@     �p@      q@     �p@     �n@     �o@     @p@     @l@     p@     �m@     �o@      n@     `k@     �j@     @f@     �i@     �j@      i@      l@     �i@     �h@     �m@     �e@     �e@      f@     @f@     �c@      d@     �f@     �e@     �e@     ��@    ���@     `i@     �k@      k@      p@     `i@      g@     `h@     `n@      l@      l@     �l@     @o@      p@     @o@      m@     @k@      o@     Pp@      o@     p@     `p@     �p@     �p@      q@     pp@     �r@      o@     @q@     �p@     pq@     �r@     q@     �r@     �q@     �t@      t@     `t@     @s@     �s@     t@     @u@     `v@     �q@     �v@     �t@     �u@     w@     �u@     �u@     �u@     �v@     �u@     pt@     py@     �w@      x@     P{@     `z@     pv@     �x@     �{@     px@     �z@      {@     �z@     z@     @~@     �y@     �{@     @}@     �{@     �{@      |@      }@     p~@     �|@     `|@     @~@     �~@     p@     �~@     ؀@     ��@     h�@     ��@     H�@     ��@      @     ��@     Ѐ@     Ё@     ��@     ��@     ��@     `�@     x�@     ��@     ��@     ��@     0�@     ��@     x�@     h�@      �@     ؇@      �@     ��@     ��@     ��@     @�@     ��@     0�@     8�@     ��@     �@     ȋ@     X�@     8�@     �@     x�@     Џ@     X�@     x�@     x�@     p�@     ��@     �@     $�@     ��@     ��@     ؓ@     ��@     �@     ��@     �@     ��@     \�@     ��@     ��@     ��@     ��@     ̙@     x�@     d�@     �@     4�@     Ȟ@     ��@     ��@     �@     �@     p�@     ֡@      �@     �@     \�@     �@     v�@     X�@     ��@     �@     ̦@     ��@     �@     $�@     �@     �@     �@     ��@     �@     �@     ��@     ��@     �@     �@     �@     #�@     "�@     z�@     ��@     �@     ��@     p�@     ȴ@     µ@     ��@     j�@     J�@     �@     �@     ��@     n�@     I�@     ��@     �@     N�@     ��@     '�@     ��@     ��@     ��@     X�@     8�@     z�@    ���@    �)�@    �u�@    ���@     �@    �\�@     ��@     =�@    �q�@     5�@    ���@    �~�@    ���@    ��@    ��@     ��@     k�@    ���@    ���@     ��@    ��@    �&�@    ���@    ��@     ��@     
�@    �l�@    ���@    ���@    ���@    ���@     ,�@    ���@    @��@     �@    ���@    �7�@    ���@    ���@    @@�@    ���@     ��@    �*�@    @J�@    ��@     5�@     7�@    ���@    ���@    �k�@    ���@     ��@     T�@    �F�@     ��@    @��@    ���@    @H�@    �h�@    ���@    ���@    ���@    @��@    @��@    ���@    �8�@    @l�@    ���@    @��@    @Z�@     ��@    �<�@    @]�@     ��@    ���@    @��@    @|�@     \�@    ���@     &�@    ���@    �f�@     ��@     ��@    ���@    0��@    �f�@    ��@    ���@    ���@    ���@     ��@     �@     �@     Ĥ@     ��@     p�@      �@        
�
predictions*�	   �~E��    �A@     ί@!  �D�7@)oZ����*@2�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x�������6�]���O�ʗ�����Zr[v��I��P=���iD*L�پ�_�T�l׾�
�%W����ӤP����iD*L��>E��a�W�>a�Ϭ(�>8K�ߝ�>�h���`�>�ߊ4F��>})�l a�>O�ʗ��>>�?�s��>�FF�G ?��[�?1��a˲?6�]��?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?w`<f@�6v��@�������:�              �?              @       @      @      "@      "@      3@      3@      ;@      4@      @@      A@      C@      D@      F@     �C@     �I@     �G@     �H@      I@     �F@     �G@      F@     �D@      F@     �J@     �D@      C@      @@     �E@      B@      ;@      8@      9@      8@      2@      9@      2@      4@      "@      &@      2@      &@      $@       @      @      $@      "@      @      @      @      @      @      @      @      @      @      @               @      @      �?      @      @      �?      @       @               @       @      @              �?              �?              �?              �?      �?              �?              �?              �?              �?      �?              �?              �?      �?      �?              �?              �?      �?               @      �?      �?      �?      �?              �?      �?      �?      �?       @       @      @      @      @      @      @      @      @      @      "@      @      @      @      @      @      @      ,@      "@      *@      3@      6@      (@      0@      4@      :@      8@      @@     �D@     �E@      I@      F@      L@     �Q@      S@      X@     �Y@     ``@     �a@     �f@     @f@     �i@     �e@      f@     �`@     �U@     �J@     �H@     �F@     �B@      8@      :@      7@      ,@      (@      .@      &@       @      @       @       @      @      @      @      (@      @      @      @       @      @      @       @       @               @       @       @      �?              �?              �?              �?              �?      �?       @              �?              �?        .�3      ���+	���i���A.*�e

mean squared errorp[%=

	r-squared\� >
�L
states*�L	      �      @   �$[RA!.Δ�
��)�,d�.A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             s�@     �@     ��@     y�@     ��@     ,�@    �4�@    ���@    ���@    ���@     v�@     ��@    �>�@    `�@    @��@     G�@    `�@    @}�@    ���@    �_�@    @<�@     n�@    @��@     ��@    ���@    ���@    @��@    ��@    ���@     �@    @��@     8�@    �f�@    �%�@     2�@    ���@     ��@    �;�@    ���@    @��@    @��@     ��@     ��@     ��@     �@    ���@    ���@    @��@    @��@    @��@    �;�@    @��@    @f�@    ���@     ��@    @��@     ��@     1�@    @��@    ���@    ��@    �G�@    ���@     .�@     ��@     R�@    @�@    ���@    @�@    @]�@    ���@    �.�@    ���@     [�@    ���@     &�@    �L�@     ��@    ���@    �5�@     E�@     Z�@    �.�@     ��@     �@    ���@    �?�@    ���@    ��@     2�@     ��@     �@     u�@     ��@    ���@     �@     >�@     -�@     �@     Ȼ@     ��@     ��@     ��@     ��@     ��@     �@     ��@     ��@     ҵ@     �@     ��@     ��@     l�@     ʳ@     +�@     ²@     ��@     ��@     ,�@     z�@     د@     ��@     ^�@     ��@     \�@     R�@     F�@     �@     ��@     2�@     ��@     �@     �@     F�@     >�@     (�@     H�@     Ȣ@     8�@     ��@     <�@     ��@     ��@     ��@     x�@     
�@      �@     ��@     ��@     ,�@     @�@     <�@     ��@     ��@     <�@     �@     ��@     D�@     l�@     D�@     ̒@     h�@     ؒ@     x�@     �@     ��@     ��@     @�@     @�@     ��@     H�@     t�@     ��@     ��@     �@     �@     0�@     ��@     ��@      �@     �@     ��@     p�@     ��@     ��@     ��@     @�@     ��@     ��@     P�@     X�@     X�@     (�@     `�@     ��@     ��@     ��@     �@     0�@     ��@     x�@     0�@     ��@     ��@     p�@     X�@     �@     ��@     �@     �@     �~@     `~@     �|@     �}@     �{@     �|@     p{@     z@     �|@     0{@     �y@     �y@     �y@     �w@      |@     z@     �x@     �y@     �z@      x@     �z@     @y@     �x@     Pv@     w@     �w@     �u@     `u@      v@     @v@     �u@     t@     �s@     @s@     @s@     @t@     �q@     �s@     s@     @s@     t@     �u@     �p@     �t@     @q@     �t@      o@     r@     �r@     �o@     `o@     Pq@     �n@     @o@     �p@      n@     �n@     `m@     �o@     �m@     �n@      j@     �o@     �m@     �n@     `j@     �o@     �l@     �o@     Pp@     �l@     �j@     �k@      j@     �i@     �h@     �h@     �g@      g@     �h@     `j@     �i@     �i@     �h@      g@     �h@      f@     �e@     @f@      d@      d@     `d@    ��@     y�@     �l@     �i@     @k@     �k@      j@      j@     �j@      h@     �j@      j@      j@     �h@     �i@     �l@      k@     @o@     �n@      l@     �k@     �o@     �o@     �m@      o@     �m@     �p@     �n@     �p@     p@     �r@     �q@     0r@     �n@     @o@     `q@     �q@     �s@     �s@      s@     �s@     �s@     �u@     pt@     Pu@     `u@     �t@     0v@     �t@     �t@     �t@     �u@     `t@      t@     �u@     Pw@     pu@     @u@     �v@     �w@     x@     @w@     �x@     �x@     py@      {@     y@     px@     �y@     �{@     �~@     P{@     @|@     �}@     @{@      {@     @y@     @z@     pz@     �{@     �|@     �{@     �}@     P}@     0|@     �|@     h�@     (�@      @     �@     0�@     @�@     x�@     �@     (�@     ��@     x�@     P�@     X�@     8�@     ��@     Є@     �@     ؄@     `�@      �@     x�@      �@     �@     X�@     ��@     `�@     h�@     �@     ��@     H�@     P�@     ��@     �@     H�@     h�@     �@     ��@     ��@     P�@     ؍@     0�@     �@     0�@     8�@     8�@     �@      �@     T�@     8�@     <�@     t�@      �@     4�@     X�@     �@     ؔ@     |�@     ��@     ̖@     ��@     ��@     D�@     �@     p�@     T�@     ��@     ,�@     D�@     0�@     h�@     d�@     ��@      @     ��@     �@     ��@     ��@     h�@     8�@     �@     Ԥ@     ��@     p�@     �@     ֦@     ��@     ��@     �@     6�@     ��@     f�@     ��@     ��@     �@     �@     M�@     T�@     f�@     ��@     ��@     2�@     �@     ��@     ��@     ��@     Y�@     ��@     �@     ��@     Q�@     �@     �@     N�@     ��@     W�@     {�@     ܼ@     ��@     q�@     ��@     ��@    �t�@    ���@    �P�@     ��@     n�@    �(�@    ��@    �(�@    �7�@    ���@     A�@    ���@    �@�@    ���@    �8�@     ��@     ��@    �+�@     ��@    ���@     H�@    �v�@     �@    ��@     .�@    ���@     ��@    @.�@    @��@    ���@     ��@    ���@     A�@    @��@     s�@     B�@    @��@    �O�@    @��@    �2�@    ���@    �@�@    ���@     ��@    ���@    @�@    ���@    @��@     t�@    ���@    �\�@    �)�@    @
�@    @��@    ��@     �@    @N�@    �*�@    ��@    @�@    ��@    �1�@    �B�@     ��@     ��@    @�@    @j�@    ���@    �/�@     t�@    ���@    @��@    @V�@    �+�@    ���@    @��@    ���@     ��@    ��@    ���@    @u�@    ���@    `x�@    ���@    `x�@    @��@    ���@     ��@     ��@     ��@     6�@     ?�@     ��@     \�@     ��@     ��@     0�@        
�
predictions*�	   ��f��    ��@     ί@!  ��H@)��d5�F@2���(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9��T7����5�i}1�x?�x��>h�'��6�]���1��a˲�E��a�Wܾ�iD*L�پ��~]�[Ӿjqs&\�Ѿ��Zr[v�>O�ʗ��>����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?������?�iZ�?+�;$�?cI���?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�              �?               @      $@      5@      2@      ?@      >@      >@     �I@      I@     �D@      M@      I@      F@     �I@      J@     �E@      G@     �O@      N@     �D@      D@      ?@      A@      B@      ?@      >@      A@      =@      >@      6@      4@      (@      2@      &@      4@      0@      $@      "@      (@      "@      7@      "@      .@      @       @      @      @      "@      @       @      @      @      @              @       @       @      @              �?       @      �?      �?      �?      @       @       @      �?       @              @      �?              �?       @              �?              �?              �?              �?              �?              �?      �?              �?               @      �?              �?      @      �?      �?      �?       @       @       @              @      @       @      @      @      @      @       @      @      @      @      ,@      @      $@      $@      @      0@      &@      &@      5@      4@      8@      =@     �B@     �D@     �C@     �F@      P@     @P@     �S@     �W@     �X@     @^@     @^@      a@     �`@     �e@     @a@      [@      [@     �[@     �N@      R@     �P@      F@      F@     �C@     �F@     �C@      :@      :@      8@      9@      3@      5@      .@      8@      *@      (@      $@      &@      "@      @      @      @      @      @      @      @      @      @               @      �?      @       @              �?              �?      �?      �?              �?      �?              �?               @        $-�	�3      EN�	��j���A/*�g

mean squared error{&=

	r-squaredh@>
�L
states*�L	   ����      @   �$[RA!��FX$��)^��y�J.A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             Ȳ@     &�@     ��@     ��@     Ż@     c�@     ��@      �@     ��@     "�@    �[�@    ���@    ��@    ���@     [�@    @��@     ��@    �e�@    �k�@    �g�@    �F�@    ���@     ,�@    ���@    ���@    �)�@    @��@    ���@    ���@     ��@     R�@    ���@    �)�@     �@    @��@     ��@     ��@    �X�@    ���@    ���@    �{�@    @��@    ���@    � �@     ��@    �$�@    @��@     ��@    �w�@    �*�@    �D�@     ��@    @��@    ���@    @5�@    ���@    @�@    ���@    @$�@    @��@    @��@    ���@    �o�@    ���@     Q�@     �@    �a�@    ��@    �m�@     ��@    @��@    ��@     ��@    @�@     �@    �Q�@     )�@     �@    ��@    �=�@    �)�@    ���@    ���@    ���@    �=�@    �L�@    ��@     +�@     h�@    ���@     Q�@    ���@    ��@     U�@    �3�@     �@     ~�@     M�@     y�@     U�@     �@     O�@     �@     �@     ʷ@     �@     ��@     ��@     ��@     \�@     )�@     ��@     ��@     S�@     5�@     �@     #�@     +�@     ��@     e�@     L�@     b�@     �@     �@     ��@     J�@     2�@      �@     Ш@     ��@     X�@     �@     Τ@     ؤ@     ��@     �@     ��@     Т@     �@     ֡@     �@     ��@     ��@     @�@     P�@     ��@     ��@     �@     x�@     ��@     x�@     ��@     l�@     ��@     ԗ@     ,�@     ��@     T�@      �@     ��@     �@     ��@     ��@     ԑ@     �@     �@     X�@     L�@     ��@     ��@     �@     P�@     �@     ��@     ��@     p�@     `�@     P�@     ��@     Ќ@     �@     ��@     ��@     ؈@     ؇@     �@     ��@     ��@     ��@     ��@     x�@     ��@     ��@     �@     ��@     ��@     @�@     ��@     �@     x�@     x�@     P�@     0�@     Ȃ@     ��@     ��@     ؀@     p�@     ؁@      �@     `@     (�@     �@     @~@      @     �}@     �@     �|@     �~@     �z@     �}@     �{@     Py@     �|@     Pz@     @{@     0x@     p{@     �z@     `x@     �z@      x@     0w@     �w@     0x@      x@     �w@      w@     �u@     �v@     Py@     �s@     �u@     �r@     Pt@     Pt@      u@     u@     s@     �s@     �v@     �r@     �r@     �t@     �s@     �r@     �r@     �s@     0s@     Pr@     0s@     �q@     �r@     �n@     Pq@     �p@     `n@     `s@     �p@     0p@     �p@     �m@     �o@     �k@     �n@     0p@     �k@     �m@      m@     @m@      n@      k@     `m@     @n@     �m@     �m@     �i@      k@     `g@     �i@     �g@     `i@     @h@     �f@     �g@     �f@      e@     �e@     �d@     `j@     @f@     �d@    �X�@    @��@     �i@     �m@     �i@     �n@     �l@     �l@      l@     @o@     �n@     �l@     �n@     �k@     @m@      o@     �m@     @o@     �o@      n@     �q@     `q@     �p@      p@     �p@     �o@     s@     �r@     Pp@     �q@     pr@     Ps@     �q@     �p@     `s@     �o@     Ps@     �r@     0v@      u@     `t@     t@     @u@     �u@     �u@     �v@     �s@     @x@     �w@     �t@     v@     �v@     w@     w@     �x@     0w@     �v@     0x@     �z@     `x@     z@     @x@     �y@     �x@      }@     p{@     �{@     0|@     pz@     P~@     }@     `{@     z@     �}@     �|@      ~@     �@     �}@     �@      }@     X�@     @�@     `@     p~@     x�@     �@     ��@     ��@     ��@     ��@     ��@     H�@      �@     ��@     h�@     (�@     x�@     ��@     ��@     �@     ��@      �@     h�@     ��@     ��@     8�@     ��@      �@     @�@     �@     ��@     ��@     @�@     ��@     ȇ@     �@     �@     ��@     P�@     0�@     ��@     �@     ȋ@     ��@     h�@     P�@     �@     ��@     ��@     |�@     ��@     h�@     �@     ,�@     x�@     ��@     �@     \�@     �@     D�@     �@     4�@     ̔@     4�@     l�@     �@     l�@     P�@     �@     ��@     ��@     t�@     ؙ@     ��@     ��@     h�@     ��@     ��@     ��@     �@     �@     ��@     .�@     ��@     ��@     R�@     J�@     ��@     ��@     ��@     V�@     ��@     ��@     ��@     ��@     *�@     B�@     ��@     ��@     ��@     �@     ֭@     N�@     ��@     ��@     ?�@     �@     ��@     ��@     =�@     ʴ@     �@     ��@     ��@     ��@     M�@     �@     ]�@     B�@     Q�@     k�@     ��@     ��@     
�@     ѽ@     �@     ��@     `�@     ��@     ��@     ��@     9�@    ���@    ���@    ���@    ���@    ���@     �@     z�@     ��@     ��@     )�@    ���@     �@    ���@    ���@     m�@    ���@     6�@     h�@     ��@    �I�@     S�@     W�@    @��@     �@     ��@    @d�@     ��@    ���@    @��@    ���@    @u�@    @��@     ��@    �o�@    ���@    ���@     �@    ���@    ���@    ���@    ���@    @ �@    @,�@    @�@    @^�@    ��@    �-�@     �@    @��@    @��@    @��@    @��@    ���@    �M�@    @��@    @��@    ��@     (�@     �@     ��@    @�@    ���@     ��@     Z�@    �Z�@    ���@     *�@     �@     =�@     p�@    @��@    �?�@     ��@    `�@     ��@    @�@    @k�@     <�@    �U�@    ��@     S�@     ��@     ��@    �j�@     X�@     �@     c�@     F�@     4�@     l�@     �@     |�@        
�
predictions*�	   ��jȿ   �i@     ί@!  P�FD�)������=@2��@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��.����ڋ��vV�R9��T7����5�i}1���d�r�x?�x��>h�'��1��a˲���[��a�Ϭ(�>8K�ߝ�>�h���`�>pz�w�7�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>�FF�G ?��[�?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�T7��?�vV�R9?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?�E̟���?yL�����?S�Fi��?�Š)U	@u�rʭ�@�������:�               @      @      @      @      *@      8@      J@      G@      I@     @T@     @T@      W@     �P@      X@     �\@     �R@     �V@      T@     �S@     �V@     �R@      S@      S@     �N@     @S@     �M@     �P@     �R@      Q@      O@     �I@     �J@      L@     �F@     �A@      F@      A@     �C@      ;@      :@      4@      ;@      <@      2@      ,@      6@      ,@      ,@      ,@      $@      ,@      6@      "@      &@      @      @      @      @      @              @       @      @      @      @      @      @       @      @       @              �?              �?              �?              �?              @      �?      �?               @              �?              �?      �?              �?              �?      �?      �?      �?               @      �?      �?      �?      �?               @      �?      �?       @              �?       @      �?              @      @      �?      @      �?      �?      �?      �?      @      @      @      @      @      @       @      @      @      &@      (@      (@      $@      *@      &@       @      (@      0@      0@      0@      8@      8@     �@@      =@      =@      A@      B@     �F@      C@      G@     �L@     �C@      G@      I@     �C@      J@     �@@      D@      I@      H@     �E@      A@      C@      F@     �D@      ?@      ;@      3@      ,@      5@      3@      7@      0@      4@      *@      &@      *@       @      @       @      @      @      @      @      @      @              �?      @       @      @       @       @      �?       @              �?              �?      �?      �?      �?              �?      �?              �?        �g�)3      ���+	s�~j���A0*�e

mean squared error��"=

	r-squared�k->
�L
states*�L	      �      @   �$[RA!4^NN��)�DI��.A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             �@     
�@     n�@     !�@     �@     ��@     �@     ��@    ���@     e�@     $�@    @��@    ���@    �F�@    ���@    ��@    �;�@     ��@    �	�@    ��@    �<�@     ��@    �a�@     7�@     _�@    ��@     �@    @/�@     �@    @"�@     W�@     �@    @��@    ���@    ���@     	�@     N�@    ���@    @R�@    @��@    @��@    �q�@    @��@     ��@    @��@     V�@     �@    @�@    @��@      �@    ���@    �y�@    @��@    ���@    @��@    ��@    @��@    @,�@    ���@    �,�@    ���@    @�@    @��@    ���@    @��@    @��@    ���@    �H�@    @��@     `�@    �$�@    ���@    ���@    �-�@     S�@    ���@     ��@    ��@     ��@    ���@     ��@    ���@    �?�@    �8�@    ���@    ���@     ��@    �*�@    �o�@    ��@    �N�@    ���@    �A�@    ���@    ��@     7�@     �@     M�@     ;�@     N�@     ܺ@     ]�@     ո@     `�@     �@     Ŷ@     ��@     ĵ@     �@     �@     �@     E�@     V�@     ��@     ��@     ��@     �@     ��@     X�@     @�@     ~�@     ¬@     Ϋ@     ҫ@     ��@     ��@     ��@     6�@     ��@     ��@     2�@     ��@     ��@     �@     `�@     H�@     ��@     &�@     ��@     ��@     l�@      �@     ��@     ��@     X�@     ��@     H�@     ��@     ��@     ��@     <�@     ��@     \�@     ��@     X�@     ��@     ܔ@     x�@      �@     Ȓ@     d�@     ܒ@     ؑ@     ��@     �@     ̑@     ��@     �@     ��@     p�@     �@     ��@     ��@     0�@     H�@     ��@     �@     �@     `�@     �@     ��@     X�@     Ј@     �@     ��@     `�@     ��@     @�@     �@     ��@     Ѕ@     ��@     ��@     �@     ��@     Є@     ��@     ��@      �@     �@     �@     ؂@     Ȃ@     ��@     ��@     �@     ��@     0�@     �@     P~@     @     ��@     @      �@     �~@     �@     `z@     �~@     `~@     �}@     0|@     p|@     P|@     `z@     �|@     pz@     �y@     py@     0w@     0w@     �x@     x@     �u@     `x@     �u@      w@     �v@     0u@     �u@     �w@      x@     �s@     �u@      s@      t@     �r@     �q@     �s@      r@     `r@     @t@     �t@     t@      p@     �p@      s@     `r@     `q@     @q@     �q@     �p@     �p@     0p@      o@     �p@      o@     pp@     �n@     `o@     �n@     �n@      o@     `n@     �m@     `l@     �l@      h@     �j@     �k@      k@     �j@     @i@     @l@     �i@      k@     �i@      k@      h@     �e@     @e@     �h@     �c@     `h@      h@     �e@      c@     �h@     �d@     �c@     �d@     @e@     �e@    ��@    �`�@     �j@      k@     �j@     @l@     �j@      h@      l@     �k@     p@      p@     �j@     �k@      l@      k@      n@     �p@     pp@     �p@     �q@     Pp@     �o@      o@     �p@      q@      p@     �o@     �q@      r@     �q@     t@     �q@     ps@     �r@     pq@     �r@      s@     �s@     �r@      q@     0s@     `s@     @t@     v@     `t@     `u@     �t@      u@     0v@     pv@     �s@     0v@     �x@     0u@     v@     �y@     �v@     �y@     �x@     `z@     �z@     @x@     �w@     �x@     �y@     P|@     `y@     �y@     �{@     0y@     �y@      {@     �y@     }@     �z@     �}@      |@     �|@     �{@     �~@     `}@     0@     ��@     �@     �~@     @~@     �@     ��@     �~@     ��@     ��@     x�@     ��@     ��@     �@     �@     x�@     ��@     �@     Ȅ@     `�@     ��@     ��@     ��@     ��@     0�@     X�@     �@     (�@     ��@     ��@     ��@     ��@     8�@     ȇ@     x�@     ��@     ��@     ��@     8�@     ��@     ȋ@     @�@     �@     h�@     p�@     p�@     ��@     Џ@     ܐ@     �@     �@     L�@     ��@     đ@     |�@     @�@     ȓ@     ��@     ��@     �@     $�@     ��@     �@     \�@     ��@     `�@     h�@     0�@     H�@     ��@      �@      �@     H�@     ��@     l�@     ��@     ��@     @�@      �@     
�@     (�@     f�@     ʢ@     j�@     Z�@     ��@     ��@     p�@     ��@     �@     >�@     ʧ@     ��@     �@     ��@     ��@     X�@     Ы@     ��@     �@     P�@     ��@     ]�@     �@     �@     �@     ~�@     D�@     ��@     ��@     I�@     ��@     ~�@     ��@     �@     �@     o�@     <�@     ��@     �@     �@     �@     X�@     ��@     ƿ@    ���@    ���@    ���@     ��@    �@�@    ���@    ���@    ���@    �'�@    �A�@    �r�@     ��@     c�@    ���@     s�@    �R�@    �9�@     ��@    ���@     ��@    ���@    ���@     ��@    ���@    ���@    �B�@    @��@     p�@    ���@    �_�@     2�@    ���@     ��@     3�@    ���@    �_�@    �<�@    ���@    ��@    ��@    @��@    @�@    �7�@    @��@    ���@    ���@     �@     �@    @N�@    ��@    ��@    ���@    @��@    ���@    ���@    @��@    ���@    �Y�@    @g�@     (�@    ��@    �>�@    @H�@    @l�@    �|�@     g�@    @H�@    @��@    ���@    ���@    @��@    ���@    �;�@    @��@     ��@    ��@    ���@     ��@     d�@     a�@     ��@    ��@    `��@    `a�@    ���@     �@    ���@    �_�@    ��@    ���@     �@     Ժ@     &�@     ��@     �@     ��@     ~�@        
�
predictions*�	   @e㶿    �%	@     ί@!  ���D@)���3�8@2�8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�U�4@@�$��[^:��"��S�F !�ji6�9���.����ڋ��vV�R9�>h�'��f�ʜ�7
�1��a˲���[��O�ʗ�����Zr[v����~]�[Ӿjqs&\�Ѿ;9��R�>���?�ګ>pz�w�7�>I��P=�>��[�?1��a˲?����?f�ʜ�7
?>h�'�?x?�x�?�5�i}1?�T7��?ji6�9�?�S�F !?�[^:��"?U�4@@�$?+A�F�&?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?yL�����?S�Fi��?h�5�@�Š)U	@�������:�               @              @      @      "@      .@      ,@     �B@      =@     �@@     �I@     �H@     �E@     �J@     �H@     �L@      I@      I@      Q@      O@     �J@      E@     �H@     �B@     �C@      E@     �B@     �A@     �D@     �E@      @@      :@      <@      5@      0@      8@      2@      .@      0@      1@      3@      7@      $@       @      *@      @      @      $@       @      $@       @      @      @      @      @      @      @      @       @      @       @              @       @       @              �?      �?      �?              �?      �?              �?              �?              �?              �?              �?              �?              �?              �?      �?       @               @              �?      �?      �?       @       @       @      �?      �?      �?      �?      @      @      @       @       @      "@      @      @       @      (@      @      "@      ,@      $@      &@      (@      8@      (@      8@      .@      ,@      7@      5@      C@      ?@      L@      D@     �C@      P@     �S@     @W@     �X@      Y@     @]@     �\@      ^@     �Y@     �V@     �W@     �V@     @P@     @R@     �R@      I@      Q@     �Q@      L@      I@      J@      F@      <@     �A@      =@      8@      7@      8@      2@      8@      *@      0@      @      "@      0@      "@      "@      @      @      @      @      @       @               @      @              �?      @       @               @      �?      �?      �?              �?      �?       @              �?              �?        �kl�r3      �i	ǟ�j���A1*�f

mean squared errorh�-=

	r-squared���=
�L
states*�L	      �      @   �$[RA!������)Sa�8��.A2�&�Š)U	�h�5���6v���w`<f���tM�ܔ�.�u��S�Fi���yL�������E̟����3?��|���P�1���cI���+�;$��iZ��������2g�G�A�uo�p�+Se*8�\l�9⿰1%���Z%�޿W�i�bۿ�^��h�ؿ��7�ֿ�Ca�G�Կ_&A�o�ҿ���ѿ�Z�_��ο�K?̿�@�"�ɿ�QK|:ǿyD$�ſ�?>8s2ÿӖ8��s��!��������(!�ؼ�%g�cE9��8/�C�ַ�� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
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
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�E̟���?yL�����?S�Fi��?ܔ�.�u�?��tM@w`<f@�6v��@h�5�@�Š)U	@�������:�&             ֶ@     �@     z�@     m�@     ѽ@    ���@     B�@     H�@     ��@    ���@    ���@     (�@     ��@     ��@    ���@    �^�@    `�@    ���@    ��@     �@    ���@     �@     ��@     ��@    ���@     ��@     ��@    @��@     �@    @��@    ���@    ���@    � �@    �G�@    �_�@    �n�@     ��@     ��@    ���@    @�@    ���@    @��@    ���@     [�@    @\�@     6�@    @B�@    ���@    @��@    �R�@    @?�@     h�@     �@    �~�@    ���@    �A�@    ���@     �@    @��@    �j�@    ���@     u�@    @��@    �t�@    ���@    @Z�@    ���@     e�@     ��@    �K�@    ���@    @�@    ���@     l�@     l�@     Y�@     G�@     q�@    ���@    ���@    ���@     }�@    �E�@     �@     c�@    ��@    ���@     �@    �2�@     ��@    �x�@    ��@    �:�@     [�@     ��@     ��@     ��@     '�@     ӻ@     ��@     ޹@     �@     �@     �@     ;�@     �@     O�@     ��@     ɴ@     X�@     �@     F�@     ɲ@     2�@     α@     ��@     ��@     /�@     ~�@     �@     0�@     Ы@     ��@     D�@     ȩ@     ^�@     B�@     x�@     p�@     ��@     &�@     V�@     p�@     Ң@     �@     �@     T�@     n�@     ��@     ��@     H�@     �@     ܝ@     (�@     ��@     ț@     ��@     ��@     X�@     D�@     ė@     L�@     l�@     ��@     ��@     X�@     �@     ܔ@     <�@     ��@     ��@     8�@     <�@     ��@     L�@     P�@     ��@     ��@     X�@     ,�@     P�@     �@      �@      �@     �@      �@     Ў@     ��@     P�@     P�@     ��@     �@     `�@     ��@     �@     8�@     p�@     h�@     8�@     ��@     ��@     `�@     @�@     ��@     ��@     ��@     ȅ@     ��@     (�@      �@     ��@     ��@      �@     ��@     �@     Ѓ@     ��@     ��@      �@     ��@     ��@     @�@     h�@     ��@     ��@     �@     p@     0~@     ~@      @     @@     `@     �@     �@      |@     @     Pz@      z@     �z@     @|@     �{@     P{@     �y@     �y@      y@     �x@     �x@     �z@      w@     �v@     Pw@     `u@     `t@     �u@     �s@     �t@      u@     �s@      r@     �s@     �t@     �r@     �s@     �t@     @v@     pt@     �t@     �s@     �s@     @q@     �p@     �r@      s@     �p@     �q@     0p@     �q@     �q@      p@     `q@     �l@     @q@     @n@     �p@     `k@     �m@     @o@      o@     Pp@     �m@     �k@      k@     �k@     `m@     `l@      k@     �i@     �f@      h@     `h@      h@     �e@     �j@     @f@     `i@     �f@     @d@     `f@     �e@     @f@     �h@     �f@    ���@    ���@     `k@     `k@     `j@      k@     �l@     �n@     `l@     @l@     �k@     p@     @l@     �n@     `m@      m@     �p@     pr@     �o@     0p@     @q@     p@     pp@      s@     Pr@     �p@      q@     �o@     �r@     �p@     �p@     �q@     pr@     0s@     �p@     �r@     �p@      s@     pr@     `s@     �r@     pt@     `s@      y@     �t@     pu@      w@     `u@     �v@     �w@     �v@     �v@     Py@     �z@     0y@     �z@     �y@     �x@     �w@     �w@     �y@     `y@     �y@     Py@     �x@      {@     y@     �y@     @y@     �y@     |@     �|@     �~@     �~@     �{@     `}@     `~@     �@      ~@     0�@     �}@     �@      |@     �@     �@     ȁ@     �@     ��@      �@     p�@     ��@     H�@     ��@     ؁@     h�@     ��@     ��@     Ȃ@     ��@     ��@     8�@     ȃ@     @�@     8�@     ��@     0�@     0�@     �@     X�@      �@      �@      �@     Ȉ@     h�@     `�@     ��@      �@     ��@     h�@     ؉@     P�@     ��@     8�@     P�@     Ѝ@     ��@     ،@     ��@     ��@     ��@     ��@     (�@     d�@     $�@     <�@     0�@     ��@     ��@     (�@     ��@     ��@     ��@     L�@     ��@     $�@     x�@     �@     ��@     ��@     ��@     �@     (�@     ��@     ��@     ��@     ��@     ��@     @�@     ��@     |�@     p�@     �@     ��@     8�@     R�@     ��@     p�@     H�@     &�@     F�@     ��@      �@     إ@     �@     ܦ@     �@     ި@     ��@     Ԫ@     �@     �@     \�@     r�@     �@     �@     ɰ@     �@     ��@     p�@     ��@     ̳@     ѳ@     @�@     \�@     �@     ڶ@     �@      �@     ��@     ��@     ��@     4�@     x�@     �@     ��@     ��@     �@     ]�@    �X�@    ���@    �z�@    ���@     �@     ��@     ��@     ~�@    ���@     L�@    ���@    ���@     �@    ���@     -�@    ���@     ��@    �c�@     V�@     �@     ��@     ��@     ��@     ��@    �;�@     ��@    �]�@     �@    @g�@    ���@    ���@     k�@    @��@    ���@     ��@    ���@    ���@     D�@     %�@    ���@    �2�@    @��@    @�@    �:�@    ���@    @��@    @��@    @��@    �y�@     ��@    @��@    �e�@     a�@    ���@    @U�@    �>�@    ��@     L�@      �@    ��@    ���@     �@    ���@    ��@    @q�@    ���@    @��@    �1�@    �z�@    ���@    ���@     u�@     {�@     ��@    ���@     8�@    ���@    �<�@    `P�@    ���@    ���@     ��@    ���@     i�@    ���@     ��@    �U�@     ��@    ���@     �@     ��@     ղ@     �@     ��@     f�@     ԥ@        
�
predictions*�	   ���ѿ    J$�?     ί@!  q6q�<@)/�C�_�(@2�_&A�o�ҿ���ѿ� l(����{ �ǳ����]$A鱿����iH��I�������g�骿�g���w���/�*>��`��a�8���uS��a���/����v��ab����<�A���}Y�4j���"�uԖ�^�S�����Rc�ݒ����&���#�h/���7c_XY��#�+(�ŉ�eiS�m��-Ա�L�����J�\������=���>	� �����T}�o��5sz�*QH�x�&b՞
�u�hyO�s�uWy��r�;8�clp��N�W�m�ߤ�(g%k�P}���h�Tw��Nof�5Ucv0ed����%��b��l�P�`�E��{��^��m9�H�[���bB�SY�ܗ�SsW�<DKc��T��lDZrS�nK���LQ�k�1^�sO�IcD���L��qU���I�
����G�a�$��{E��T���C��!�A����#@�d�\D�X=���%>��:�uܬ�@8���%�V6��u�w74���82���bȬ�0���VlQ.��7Kaa+�I�I�)�(�+A�F�&�U�4@@�$��[^:��"��S�F !���d�r�x?�x��>h�'��f�ʜ�7
�������FF�G �>�?�s���O�ʗ���8K�ߝ�a�Ϭ(�jqs&\�ѾK+�E��Ͼ;�"�qʾ
�/eq
Ⱦ����ž�XQ�þ��>M|K�>�_�T�l�>�ѩ�-�>���%�>I��P=�>��Zr[v�>O�ʗ��>>�?�s��>1��a˲?6�]��?����?f�ʜ�7
?>h�'�?x?�x�?��d�r?�5�i}1?��ڋ?�.�?ji6�9�?�S�F !?�[^:��"?I�I�)�(?�7Kaa+?��VlQ.?��bȬ�0?��82?�u�w74?��%�V6?uܬ�@8?��%>��:?d�\D�X=?���#@?�!�A?�T���C?a�$��{E?
����G?�qU���I?IcD���L?k�1^�sO?nK���LQ?�lDZrS?<DKc��T?ܗ�SsW?��bB�SY?�m9�H�[?E��{��^?�l�P�`?���%��b?5Ucv0ed?Tw��Nof?P}���h?ߤ�(g%k?�N�W�m?;8�clp?uWy��r?hyO�s?&b՞
�u?*QH�x?o��5sz?���T}?>	� �?����=��?���J�\�?-Ա�L�?eiS�m�?#�+(�ŉ?�7c_XY�?�#�h/�?���&�?�Rc�ݒ?^�S���?�"�uԖ?}Y�4j�?��<�A��?�v��ab�?�/��?�uS��a�?`��a�8�?�/�*>�?�g���w�?���g��?I���?����iH�?��]$A�?�{ �ǳ�?� l(��?8/�C�ַ?%g�cE9�?��(!�ؼ?!�����?Ӗ8��s�?�?>8s2�?yD$��?�QK|:�?�@�"��?�K?�?�Z�_���?����?_&A�o��?�Ca�G��?��7��?�^��h��?W�i�b�?��Z%��?�1%�?\l�9�?+Se*8�?uo�p�?2g�G�A�?������?�iZ�?+�;$�?cI���?�P�1���?3?��|�?�������:�              �?              �?      @      $@      6@      =@      :@     �F@      G@     �Q@     @P@     �U@     �R@     @R@      R@     @T@     �U@     �Q@      P@     @Q@     �M@      N@      J@     �F@     �M@     �D@      C@      I@      F@     �J@     �E@      8@      @@      A@      4@      <@      =@      ?@      7@      .@      ,@      0@      .@      $@      @      (@      "@      @      @      $@      @      @      @      @      @      @       @      @      @              @              @       @      @      �?      @      �?      @               @       @              �?              �?      �?              �?              �?              �?              �?              �?              �?               @              �?              �?      �?       @      �?              �?      @              �?      �?              @               @       @      @       @      @      �?      @      @       @      @      @      @      @      @       @      *@       @      (@      *@      "@      3@      &@      2@      3@      4@      ,@      1@     �B@      ;@      C@     �C@      C@     �E@      F@      F@      I@      N@     �P@     �P@      Q@     �P@     �O@      N@     �Q@      N@     �I@     �J@     �M@      I@      D@      L@      H@      K@      F@     �D@      ?@     �C@      5@      :@      A@      8@      6@      6@      5@      .@      7@      9@      0@      0@      0@      ,@      (@      @      @      @      @      @      @      @      @      �?      �?              �?              �?              �?      �?              �?              �?        h_�