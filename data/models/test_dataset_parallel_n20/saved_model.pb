ΟΘ
Υͺ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018Φ
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ΐ*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	ΐ*
dtype0

!Adam/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_2/beta/v

5Adam/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/v*
_output_shapes
: *
dtype0

"Adam/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_2/gamma/v

6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/v*
_output_shapes
: *
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes
:@*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:*
dtype0

Adam/conv1d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_2/bias/v
y
(Adam/conv1d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/v*
_output_shapes
: *
dtype0

Adam/conv1d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *'
shared_nameAdam/conv1d_2/kernel/v

*Adam/conv1d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/v*"
_output_shapes
:
 *
dtype0

Adam/conv1d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_1/bias/v
y
(Adam/conv1d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/v*
_output_shapes
:@*
dtype0

Adam/conv1d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*'
shared_nameAdam/conv1d_1/kernel/v

*Adam/conv1d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/v*"
_output_shapes
:
@*
dtype0
}
Adam/conv1d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/v
v
&Adam/conv1d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/v*
_output_shapes	
:*
dtype0

Adam/conv1d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d/kernel/v

(Adam/conv1d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/v*#
_output_shapes
:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ΐ*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	ΐ*
dtype0

!Adam/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/batch_normalization_2/beta/m

5Adam/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_2/beta/m*
_output_shapes
: *
dtype0

"Adam/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_2/gamma/m

6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_2/gamma/m*
_output_shapes
: *
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes
:@*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes
:@*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:*
dtype0

Adam/conv1d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv1d_2/bias/m
y
(Adam/conv1d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/bias/m*
_output_shapes
: *
dtype0

Adam/conv1d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *'
shared_nameAdam/conv1d_2/kernel/m

*Adam/conv1d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_2/kernel/m*"
_output_shapes
:
 *
dtype0

Adam/conv1d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv1d_1/bias/m
y
(Adam/conv1d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/bias/m*
_output_shapes
:@*
dtype0

Adam/conv1d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*'
shared_nameAdam/conv1d_1/kernel/m

*Adam/conv1d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d_1/kernel/m*"
_output_shapes
:
@*
dtype0
}
Adam/conv1d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv1d/bias/m
v
&Adam/conv1d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/bias/m*
_output_shapes	
:*
dtype0

Adam/conv1d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv1d/kernel/m

(Adam/conv1d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1d/kernel/m*#
_output_shapes
:
*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ΐ*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	ΐ*
dtype0
’
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0
’
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:*
dtype0
r
conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv1d_2/bias
k
!conv1d_2/bias/Read/ReadVariableOpReadVariableOpconv1d_2/bias*
_output_shapes
: *
dtype0
~
conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 * 
shared_nameconv1d_2/kernel
w
#conv1d_2/kernel/Read/ReadVariableOpReadVariableOpconv1d_2/kernel*"
_output_shapes
:
 *
dtype0
r
conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv1d_1/bias
k
!conv1d_1/bias/Read/ReadVariableOpReadVariableOpconv1d_1/bias*
_output_shapes
:@*
dtype0
~
conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@* 
shared_nameconv1d_1/kernel
w
#conv1d_1/kernel/Read/ReadVariableOpReadVariableOpconv1d_1/kernel*"
_output_shapes
:
@*
dtype0
o
conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1d/bias
h
conv1d/bias/Read/ReadVariableOpReadVariableOpconv1d/bias*
_output_shapes	
:*
dtype0
{
conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv1d/kernel
t
!conv1d/kernel/Read/ReadVariableOpReadVariableOpconv1d/kernel*#
_output_shapes
:
*
dtype0

NoOpNoOp
Υ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bψ
τ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
Θ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op*
Θ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op*
Θ
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op*
Υ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*
Υ
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance*
Υ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance*
₯
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator* 
₯
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator* 
₯
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator* 

m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 

s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses* 

y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*

"0
#1
+2
,3
44
55
>6
?7
@8
A9
I10
J11
K12
L13
T14
U15
V16
W17
18
19*
l
"0
#1
+2
,3
44
55
>6
?7
I8
J9
T10
U11
12
13*
* 
΅
non_trainable_variables
 layers
‘metrics
 ’layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
€trace_0
₯trace_1
¦trace_2
§trace_3* 
:
¨trace_0
©trace_1
ͺtrace_2
«trace_3* 
* 
ε
	¬iter
­beta_1
?beta_2

―decay
°learning_rate"mΐ#mΑ+mΒ,mΓ4mΔ5mΕ>mΖ?mΗImΘJmΙTmΚUmΛ	mΜ	mΝ"vΞ#vΟ+vΠ,vΡ4v?5vΣ>vΤ?vΥIvΦJvΧTvΨUvΩ	vΪ	vΫ*

±serving_default* 

"0
#1*

"0
#1*
* 

²non_trainable_variables
³layers
΄metrics
 ΅layer_regularization_losses
Άlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

·trace_0* 

Έtrace_0* 
]W
VARIABLE_VALUEconv1d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv1d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

+0
,1*

+0
,1*
* 

Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Ύtrace_0* 

Ώtrace_0* 
_Y
VARIABLE_VALUEconv1d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

40
51*

40
51*
* 

ΐnon_trainable_variables
Αlayers
Βmetrics
 Γlayer_regularization_losses
Δlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses*

Εtrace_0* 

Ζtrace_0* 
_Y
VARIABLE_VALUEconv1d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv1d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
>0
?1
@2
A3*

>0
?1*
* 

Ηnon_trainable_variables
Θlayers
Ιmetrics
 Κlayer_regularization_losses
Λlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

Μtrace_0
Νtrace_1* 

Ξtrace_0
Οtrace_1* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
I0
J1
K2
L3*

I0
J1*
* 

Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

Υtrace_0
Φtrace_1* 

Χtrace_0
Ψtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
T0
U1
V2
W3*

T0
U1*
* 

Ωnon_trainable_variables
Ϊlayers
Ϋmetrics
 άlayer_regularization_losses
έlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*

ήtrace_0
ίtrace_1* 

ΰtrace_0
αtrace_1* 
* 
jd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

βnon_trainable_variables
γlayers
δmetrics
 εlayer_regularization_losses
ζlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses* 

ηtrace_0
θtrace_1* 

ιtrace_0
κtrace_1* 
* 
* 
* 
* 

λnon_trainable_variables
μlayers
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses* 

πtrace_0
ρtrace_1* 

ςtrace_0
σtrace_1* 
* 
* 
* 
* 

τnon_trainable_variables
υlayers
φmetrics
 χlayer_regularization_losses
ψlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses* 

ωtrace_0
ϊtrace_1* 

ϋtrace_0
όtrace_1* 
* 
* 
* 
* 

ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

₯trace_0* 

¦trace_0* 
* 
* 
* 

§non_trainable_variables
¨layers
©metrics
 ͺlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

¬trace_0* 

­trace_0* 

0
1*

0
1*
* 

?non_trainable_variables
―layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

³trace_0* 

΄trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
@0
A1
K2
L3
V4
W5*

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

΅0
Ά1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 

K0
L1*
* 
* 
* 
* 
* 
* 
* 
* 

V0
W1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
·	variables
Έ	keras_api

Ήtotal

Ίcount*
M
»	variables
Ό	keras_api

½total

Ύcount
Ώ
_fn_kwargs*

Ή0
Ί1*

·	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

½0
Ύ1*

»	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
z
VARIABLE_VALUEAdam/conv1d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv1d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv1d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv1d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv1d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/batch_normalization/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/batch_normalization/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_1/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/batch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*+
_output_shapes
:?????????
*
dtype0* 
shape:?????????

Σ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1d_2/kernelconv1d_2/biasconv1d_1/kernelconv1d_1/biasconv1d/kernelconv1d/bias%batch_normalization_2/moving_variancebatch_normalization_2/gamma!batch_normalization_2/moving_meanbatch_normalization_2/beta%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/beta#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betadense/kernel
dense/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_42882
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
€
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv1d/kernel/Read/ReadVariableOpconv1d/bias/Read/ReadVariableOp#conv1d_1/kernel/Read/ReadVariableOp!conv1d_1/bias/Read/ReadVariableOp#conv1d_2/kernel/Read/ReadVariableOp!conv1d_2/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv1d/kernel/m/Read/ReadVariableOp&Adam/conv1d/bias/m/Read/ReadVariableOp*Adam/conv1d_1/kernel/m/Read/ReadVariableOp(Adam/conv1d_1/bias/m/Read/ReadVariableOp*Adam/conv1d_2/kernel/m/Read/ReadVariableOp(Adam/conv1d_2/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_2/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv1d/kernel/v/Read/ReadVariableOp&Adam/conv1d/bias/v/Read/ReadVariableOp*Adam/conv1d_1/kernel/v/Read/ReadVariableOp(Adam/conv1d_1/bias/v/Read/ReadVariableOp*Adam/conv1d_2/kernel/v/Read/ReadVariableOp(Adam/conv1d_2/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp6Adam/batch_normalization_2/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_2/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*F
Tin?
=2;	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_43965
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1d/kernelconv1d/biasconv1d_1/kernelconv1d_1/biasconv1d_2/kernelconv1d_2/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancebatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancebatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variancedense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv1d/kernel/mAdam/conv1d/bias/mAdam/conv1d_1/kernel/mAdam/conv1d_1/bias/mAdam/conv1d_2/kernel/mAdam/conv1d_2/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/m"Adam/batch_normalization_2/gamma/m!Adam/batch_normalization_2/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv1d/kernel/vAdam/conv1d/bias/vAdam/conv1d_1/kernel/vAdam/conv1d_1/bias/vAdam/conv1d_2/kernel/vAdam/conv1d_2/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/v"Adam/batch_normalization_2/gamma/v!Adam/batch_normalization_2/beta/vAdam/dense/kernel/vAdam/dense/bias/v*E
Tin>
<2:*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_44146ϊ?

±
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41891

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:??????????????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:??????????????????p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:??????????????????Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
Ύ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
Ξ

A__inference_conv1d_layer_call_and_return_conditional_losses_42225

inputsB
+conv1d_expanddims_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:
*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ‘
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:
­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

ύ????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
L
χ	
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42619

inputs$
conv1d_2_42561:
 
conv1d_2_42563: $
conv1d_1_42566:
@
conv1d_1_42568:@#
conv1d_42571:

conv1d_42573:	)
batch_normalization_2_42576: )
batch_normalization_2_42578: )
batch_normalization_2_42580: )
batch_normalization_2_42582: )
batch_normalization_1_42585:@)
batch_normalization_1_42587:@)
batch_normalization_1_42589:@)
batch_normalization_1_42591:@(
batch_normalization_42594:	(
batch_normalization_42596:	(
batch_normalization_42598:	(
batch_normalization_42600:	
dense_42613:	ΐ
dense_42615:
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCallτ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_42561conv1d_2_42563*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181τ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_42566conv1d_1_42568*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203ν
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_42571conv1d_42573*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_42225
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_42576batch_normalization_2_42578batch_normalization_2_42580batch_normalization_2_42582*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42102
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_42585batch_normalization_1_42587batch_normalization_1_42589batch_normalization_1_42591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_42020ϊ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_42594batch_normalization_42596batch_normalization_42598batch_normalization_42600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41938
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42478€
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42455
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42432π
max_pooling1d_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155π
max_pooling1d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140λ
max_pooling1d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125Ω
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_42288ί
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296ί
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304₯
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_42314
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_42613dense_42615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ε
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
ύό

T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43269

inputsJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:
 6
(conv1d_2_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:
@6
(conv1d_1_biasadd_readvariableop_resource:@I
2conv1d_conv1d_expanddims_1_readvariableop_resource:
5
&conv1d_biasadd_readvariableop_resource:	K
=batch_normalization_2_assignmovingavg_readvariableop_resource: M
?batch_normalization_2_assignmovingavg_1_readvariableop_resource: I
;batch_normalization_2_batchnorm_mul_readvariableop_resource: E
7batch_normalization_2_batchnorm_readvariableop_resource: K
=batch_normalization_1_assignmovingavg_readvariableop_resource:@M
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@E
7batch_normalization_1_batchnorm_readvariableop_resource:@J
;batch_normalization_assignmovingavg_readvariableop_resource:	L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	H
9batch_normalization_batchnorm_mul_readvariableop_resource:	D
5batch_normalization_batchnorm_readvariableop_resource:	7
$dense_matmul_readvariableop_resource:	ΐ3
%dense_biasadd_readvariableop_resource:
identity’#batch_normalization/AssignMovingAvg’2batch_normalization/AssignMovingAvg/ReadVariableOp’%batch_normalization/AssignMovingAvg_1’4batch_normalization/AssignMovingAvg_1/ReadVariableOp’,batch_normalization/batchnorm/ReadVariableOp’0batch_normalization/batchnorm/mul/ReadVariableOp’%batch_normalization_1/AssignMovingAvg’4batch_normalization_1/AssignMovingAvg/ReadVariableOp’'batch_normalization_1/AssignMovingAvg_1’6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_1/batchnorm/ReadVariableOp’2batch_normalization_1/batchnorm/mul/ReadVariableOp’%batch_normalization_2/AssignMovingAvg’4batch_normalization_2/AssignMovingAvg/ReadVariableOp’'batch_normalization_2/AssignMovingAvg_1’6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp’.batch_normalization_2/batchnorm/ReadVariableOp’2batch_normalization_2/batchnorm/mul/ReadVariableOp’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_2/BiasAdd/ReadVariableOp’+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOpi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
€
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
 *
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
 Η
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

ύ????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d_1/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
@Η
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
‘
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:
*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ά
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:
Β
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Δ
"batch_normalization_2/moments/meanMeanconv1d_2/Relu:activations:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
: Μ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv1d_2/Relu:activations:03batch_normalization_2/moments/StopGradient:output:0*
T0*+
_output_shapes
:????????? 
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       δ
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
  
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 p
+batch_normalization_2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<?
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0Γ
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*
_output_shapes
: Ί
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: 
%batch_normalization_2/AssignMovingAvgAssignSubVariableOp=batch_normalization_2_assignmovingavg_readvariableop_resource-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<²
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0Ι
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*
_output_shapes
: ΐ
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: 
'batch_normalization_2/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_2_assignmovingavg_1_readvariableop_resource/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ͺ
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Ά
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: ¨
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ͺ
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ’
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0²
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: Έ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? 
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Δ
"batch_normalization_1/moments/meanMeanconv1d_1/Relu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
:@Μ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/Relu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*+
_output_shapes
:?????????@
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       δ
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
  
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<?
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0Γ
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes
:@Ί
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<²
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0Ι
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@ΐ
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:³
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ͺ
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¨
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@ͺ
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@’
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0²
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Έ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       Ώ
 batch_normalization/moments/meanMeanconv1d/Relu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*#
_output_shapes
:Η
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv1d/Relu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ί
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<«
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ύ
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:΅
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ό
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<―
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Δ
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:»
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:§
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0±
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:£
#batch_normalization/batchnorm/mul_1Mulconv1d/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????₯
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0­
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:³
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????\
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?
dropout_2/dropout/MulMul)batch_normalization_2/batchnorm/add_1:z:0 dropout_2/dropout/Const:output:0*
T0*+
_output_shapes
:????????? p
dropout_2/dropout/ShapeShape)batch_normalization_2/batchnorm/add_1:z:0*
T0*
_output_shapes
:€
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0e
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Θ
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? 
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? 
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*+
_output_shapes
:????????? \
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?
dropout_1/dropout/MulMul)batch_normalization_1/batchnorm/add_1:z:0 dropout_1/dropout/Const:output:0*
T0*+
_output_shapes
:?????????@p
dropout_1/dropout/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:€
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Θ
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?
dropout/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*,
_output_shapes
:?????????l
dropout/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:‘
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Γ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*,
_output_shapes
:?????????`
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¨
max_pooling1d_2/ExpandDims
ExpandDimsdropout_2/dropout/Mul_1:z:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? ΄
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????
 *
ksize
*
paddingVALID*
strides

max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????
 *
squeeze_dims
`
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¨
max_pooling1d_1/ExpandDims
ExpandDimsdropout_1/dropout/Mul_1:z:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@΄
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:?????????
@*
ksize
*
paddingVALID*
strides

max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????
@*
squeeze_dims
^
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :£
max_pooling1d/ExpandDims
ExpandDimsdropout/dropout/Mul_1:z:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????±
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides

max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????
*
squeeze_dims
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  
flatten_1/ReshapeReshape max_pooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  
flatten_2/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????ΐY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ξ
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????ΐ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ΐ*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Ϋ	
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp&^batch_normalization_2/AssignMovingAvg5^batch_normalization_2/AssignMovingAvg/ReadVariableOp(^batch_normalization_2/AssignMovingAvg_17^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp3^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2N
%batch_normalization_2/AssignMovingAvg%batch_normalization_2/AssignMovingAvg2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_2/AssignMovingAvg_1'batch_normalization_2/AssignMovingAvg_12p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
Ο
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
ͺ
E
)__inference_flatten_2_layer_call_fn_43731

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
 :S O
+
_output_shapes
:?????????
 
 
_user_specified_nameinputs
s
¬
__inference__traced_save_43965
file_prefix,
(savev2_conv1d_kernel_read_readvariableop*
&savev2_conv1d_bias_read_readvariableop.
*savev2_conv1d_1_kernel_read_readvariableop,
(savev2_conv1d_1_bias_read_readvariableop.
*savev2_conv1d_2_kernel_read_readvariableop,
(savev2_conv1d_2_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv1d_kernel_m_read_readvariableop1
-savev2_adam_conv1d_bias_m_read_readvariableop5
1savev2_adam_conv1d_1_kernel_m_read_readvariableop3
/savev2_adam_conv1d_1_bias_m_read_readvariableop5
1savev2_adam_conv1d_2_kernel_m_read_readvariableop3
/savev2_adam_conv1d_2_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv1d_kernel_v_read_readvariableop1
-savev2_adam_conv1d_bias_v_read_readvariableop5
1savev2_adam_conv1d_1_kernel_v_read_readvariableop3
/savev2_adam_conv1d_1_bias_v_read_readvariableop5
1savev2_adam_conv1d_2_kernel_v_read_readvariableop3
/savev2_adam_conv1d_2_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableopA
=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_2_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ξ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*χ
valueνBκ:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHβ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ζ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv1d_kernel_read_readvariableop&savev2_conv1d_bias_read_readvariableop*savev2_conv1d_1_kernel_read_readvariableop(savev2_conv1d_1_bias_read_readvariableop*savev2_conv1d_2_kernel_read_readvariableop(savev2_conv1d_2_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv1d_kernel_m_read_readvariableop-savev2_adam_conv1d_bias_m_read_readvariableop1savev2_adam_conv1d_1_kernel_m_read_readvariableop/savev2_adam_conv1d_1_bias_m_read_readvariableop1savev2_adam_conv1d_2_kernel_m_read_readvariableop/savev2_adam_conv1d_2_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=savev2_adam_batch_normalization_2_gamma_m_read_readvariableop<savev2_adam_batch_normalization_2_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv1d_kernel_v_read_readvariableop-savev2_adam_conv1d_bias_v_read_readvariableop1savev2_adam_conv1d_1_kernel_v_read_readvariableop/savev2_adam_conv1d_1_bias_v_read_readvariableop1savev2_adam_conv1d_2_kernel_v_read_readvariableop/savev2_adam_conv1d_2_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=savev2_adam_batch_normalization_2_gamma_v_read_readvariableop<savev2_adam_batch_normalization_2_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *H
dtypes>
<2:	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*°
_input_shapes
: :
::
@:@:
 : :::::@:@:@:@: : : : :	ΐ:: : : : : : : : : :
::
@:@:
 : :::@:@: : :	ΐ::
::
@:@:
 : :::@:@: : :	ΐ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:)%
#
_output_shapes
:
:!

_output_shapes	
::($
"
_output_shapes
:
@: 

_output_shapes
:@:($
"
_output_shapes
:
 : 

_output_shapes
: :!

_output_shapes	
::!

_output_shapes	
::!	

_output_shapes	
::!


_output_shapes	
:: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	ΐ: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :)%
#
_output_shapes
:
:!

_output_shapes	
::( $
"
_output_shapes
:
@: !

_output_shapes
:@:("$
"
_output_shapes
:
 : #

_output_shapes
: :!$

_output_shapes	
::!%

_output_shapes	
:: &

_output_shapes
:@: '

_output_shapes
:@: (

_output_shapes
: : )

_output_shapes
: :%*!

_output_shapes
:	ΐ: +

_output_shapes
::),%
#
_output_shapes
:
:!-

_output_shapes	
::(.$
"
_output_shapes
:
@: /

_output_shapes
:@:(0$
"
_output_shapes
:
 : 1

_output_shapes
: :!2

_output_shapes	
::!3

_output_shapes	
:: 4

_output_shapes
:@: 5

_output_shapes
:@: 6

_output_shapes
: : 7

_output_shapes
: :%8!

_output_shapes
:	ΐ: 9

_output_shapes
:::

_output_shapes
: 
Χ

&__inference_conv1d_layer_call_fn_43278

inputs
unknown:

	unknown_0:	
identity’StatefulPartitionedCallή
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_42225t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

―
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41973

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_43599

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ι
`
B__inference_dropout_layer_call_and_return_conditional_losses_42277

inputs

identity_1S
IdentityIdentityinputs*
T0*,
_output_shapes
:?????????`

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:?????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Ύ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_43737

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????ΐY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
 :S O
+
_output_shapes
:?????????
 
 
_user_specified_nameinputs


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_43638

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ͺ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
ΣG
	
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42333

inputs$
conv1d_2_42182:
 
conv1d_2_42184: $
conv1d_1_42204:
@
conv1d_1_42206:@#
conv1d_42226:

conv1d_42228:	)
batch_normalization_2_42231: )
batch_normalization_2_42233: )
batch_normalization_2_42235: )
batch_normalization_2_42237: )
batch_normalization_1_42240:@)
batch_normalization_1_42242:@)
batch_normalization_1_42244:@)
batch_normalization_1_42246:@(
batch_normalization_42249:	(
batch_normalization_42251:	(
batch_normalization_42253:	(
batch_normalization_42255:	
dense_42327:	ΐ
dense_42329:
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’dense/StatefulPartitionedCallτ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_2_42182conv1d_2_42184*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181τ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_1_42204conv1d_1_42206*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203ν
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_42226conv1d_42228*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_42225
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_42231batch_normalization_2_42233batch_normalization_2_42235batch_normalization_2_42237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42055
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_42240batch_normalization_1_42242batch_normalization_1_42244batch_normalization_1_42246*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41973ό
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_42249batch_normalization_42251batch_normalization_42253batch_normalization_42255*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41891π
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42263π
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42270λ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42277θ
max_pooling1d_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155θ
max_pooling1d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140γ
max_pooling1d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125Ω
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_42288ί
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296ί
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304₯
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_42314
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_42327dense_42329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ϋ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
Ύ
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????ΐY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
 :S O
+
_output_shapes
:?????????
 
 
_user_specified_nameinputs
η
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_42270

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ο
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_43704

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ξ

A__inference_conv1d_layer_call_and_return_conditional_losses_43294

inputsB
+conv1d_expanddims_1_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:
*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ‘
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:
­
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

ύ????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????U
ReluReluBiasAdd:output:0*
T0*,
_output_shapes
:?????????f
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_43665

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ͺ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
ό%
ι
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_42020

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ’
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Χ

(__inference_conv1d_1_layer_call_fn_43303

inputs
unknown:
@
	unknown_0:@
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
ΑΧ
΅
 __inference__wrapped_model_41867
input_1d
Ntest_dataset_parallel_n20_conv1d_2_conv1d_expanddims_1_readvariableop_resource:
 P
Btest_dataset_parallel_n20_conv1d_2_biasadd_readvariableop_resource: d
Ntest_dataset_parallel_n20_conv1d_1_conv1d_expanddims_1_readvariableop_resource:
@P
Btest_dataset_parallel_n20_conv1d_1_biasadd_readvariableop_resource:@c
Ltest_dataset_parallel_n20_conv1d_conv1d_expanddims_1_readvariableop_resource:
O
@test_dataset_parallel_n20_conv1d_biasadd_readvariableop_resource:	_
Qtest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_resource: c
Utest_dataset_parallel_n20_batch_normalization_2_batchnorm_mul_readvariableop_resource: a
Stest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_1_resource: a
Stest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_2_resource: _
Qtest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_resource:@c
Utest_dataset_parallel_n20_batch_normalization_1_batchnorm_mul_readvariableop_resource:@a
Stest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_1_resource:@a
Stest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_2_resource:@^
Otest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_resource:	b
Stest_dataset_parallel_n20_batch_normalization_batchnorm_mul_readvariableop_resource:	`
Qtest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_1_resource:	`
Qtest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_2_resource:	Q
>test_dataset_parallel_n20_dense_matmul_readvariableop_resource:	ΐM
?test_dataset_parallel_n20_dense_biasadd_readvariableop_resource:
identity’Ftest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp’Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_1’Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_2’Jtest_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOp’Htest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp’Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_1’Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_2’Ltest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOp’Htest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp’Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_1’Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_2’Ltest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOp’7test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOp’Ctest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOp’9test_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOp’Etest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’9test_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOp’Etest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’6test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOp’5test_dataset_parallel_n20/dense/MatMul/ReadVariableOp
8test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Θ
4test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims
ExpandDimsinput_1Atest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
Ψ
Etest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpNtest_dataset_parallel_n20_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
 *
dtype0|
:test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
6test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1
ExpandDimsMtest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ctest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
 
)test_dataset_parallel_n20/conv1d_2/Conv1DConv2D=test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims:output:0?test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
Ζ
1test_dataset_parallel_n20/conv1d_2/Conv1D/SqueezeSqueeze2test_dataset_parallel_n20/conv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

ύ????????Έ
9test_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOpReadVariableOpBtest_dataset_parallel_n20_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0κ
*test_dataset_parallel_n20/conv1d_2/BiasAddBiasAdd:test_dataset_parallel_n20/conv1d_2/Conv1D/Squeeze:output:0Atest_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? 
'test_dataset_parallel_n20/conv1d_2/ReluRelu3test_dataset_parallel_n20/conv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? 
8test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Θ
4test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims
ExpandDimsinput_1Atest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
Ψ
Etest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpNtest_dataset_parallel_n20_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
@*
dtype0|
:test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
6test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1
ExpandDimsMtest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0Ctest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
@
)test_dataset_parallel_n20/conv1d_1/Conv1DConv2D=test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims:output:0?test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
Ζ
1test_dataset_parallel_n20/conv1d_1/Conv1D/SqueezeSqueeze2test_dataset_parallel_n20/conv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

ύ????????Έ
9test_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOpReadVariableOpBtest_dataset_parallel_n20_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0κ
*test_dataset_parallel_n20/conv1d_1/BiasAddBiasAdd:test_dataset_parallel_n20/conv1d_1/Conv1D/Squeeze:output:0Atest_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@
'test_dataset_parallel_n20/conv1d_1/ReluRelu3test_dataset_parallel_n20/conv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@
6test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????Δ
2test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims
ExpandDimsinput_1?test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
Υ
Ctest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOpLtest_dataset_parallel_n20_conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:
*
dtype0z
8test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 
4test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1
ExpandDimsKtest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0Atest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:

'test_dataset_parallel_n20/conv1d/Conv1DConv2D;test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims:output:0=test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides
Γ
/test_dataset_parallel_n20/conv1d/Conv1D/SqueezeSqueeze0test_dataset_parallel_n20/conv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

ύ????????΅
7test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOpReadVariableOp@test_dataset_parallel_n20_conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ε
(test_dataset_parallel_n20/conv1d/BiasAddBiasAdd8test_dataset_parallel_n20/conv1d/Conv1D/Squeeze:output:0?test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????
%test_dataset_parallel_n20/conv1d/ReluRelu1test_dataset_parallel_n20/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????Φ
Htest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOpQtest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0
?test_dataset_parallel_n20/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
=test_dataset_parallel_n20/batch_normalization_2/batchnorm/addAddV2Ptest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp:value:0Htest_dataset_parallel_n20/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: °
?test_dataset_parallel_n20/batch_normalization_2/batchnorm/RsqrtRsqrtAtest_dataset_parallel_n20/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ή
Ltest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpUtest_dataset_parallel_n20_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0
=test_dataset_parallel_n20/batch_normalization_2/batchnorm/mulMulCtest_dataset_parallel_n20/batch_normalization_2/batchnorm/Rsqrt:y:0Ttest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: φ
?test_dataset_parallel_n20/batch_normalization_2/batchnorm/mul_1Mul5test_dataset_parallel_n20/conv1d_2/Relu:activations:0Atest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? Ϊ
Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpStest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0
?test_dataset_parallel_n20/batch_normalization_2/batchnorm/mul_2MulRtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0Atest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: Ϊ
Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpStest_dataset_parallel_n20_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0
=test_dataset_parallel_n20/batch_normalization_2/batchnorm/subSubRtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_2:value:0Ctest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
?test_dataset_parallel_n20/batch_normalization_2/batchnorm/add_1AddV2Ctest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul_1:z:0Atest_dataset_parallel_n20/batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? Φ
Htest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpQtest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0
?test_dataset_parallel_n20/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
=test_dataset_parallel_n20/batch_normalization_1/batchnorm/addAddV2Ptest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp:value:0Htest_dataset_parallel_n20/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@°
?test_dataset_parallel_n20/batch_normalization_1/batchnorm/RsqrtRsqrtAtest_dataset_parallel_n20/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ή
Ltest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpUtest_dataset_parallel_n20_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0
=test_dataset_parallel_n20/batch_normalization_1/batchnorm/mulMulCtest_dataset_parallel_n20/batch_normalization_1/batchnorm/Rsqrt:y:0Ttest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@φ
?test_dataset_parallel_n20/batch_normalization_1/batchnorm/mul_1Mul5test_dataset_parallel_n20/conv1d_1/Relu:activations:0Atest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@Ϊ
Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpStest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0
?test_dataset_parallel_n20/batch_normalization_1/batchnorm/mul_2MulRtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0Atest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@Ϊ
Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpStest_dataset_parallel_n20_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0
=test_dataset_parallel_n20/batch_normalization_1/batchnorm/subSubRtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0Ctest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
?test_dataset_parallel_n20/batch_normalization_1/batchnorm/add_1AddV2Ctest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul_1:z:0Atest_dataset_parallel_n20/batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@Σ
Ftest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOpReadVariableOpOtest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0
=test_dataset_parallel_n20/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
;test_dataset_parallel_n20/batch_normalization/batchnorm/addAddV2Ntest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp:value:0Ftest_dataset_parallel_n20/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:­
=test_dataset_parallel_n20/batch_normalization/batchnorm/RsqrtRsqrt?test_dataset_parallel_n20/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ϋ
Jtest_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpStest_dataset_parallel_n20_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0?
;test_dataset_parallel_n20/batch_normalization/batchnorm/mulMulAtest_dataset_parallel_n20/batch_normalization/batchnorm/Rsqrt:y:0Rtest_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:ρ
=test_dataset_parallel_n20/batch_normalization/batchnorm/mul_1Mul3test_dataset_parallel_n20/conv1d/Relu:activations:0?test_dataset_parallel_n20/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????Χ
Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpQtest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ύ
=test_dataset_parallel_n20/batch_normalization/batchnorm/mul_2MulPtest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_1:value:0?test_dataset_parallel_n20/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:Χ
Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpQtest_dataset_parallel_n20_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ύ
;test_dataset_parallel_n20/batch_normalization/batchnorm/subSubPtest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_2:value:0Atest_dataset_parallel_n20/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
=test_dataset_parallel_n20/batch_normalization/batchnorm/add_1AddV2Atest_dataset_parallel_n20/batch_normalization/batchnorm/mul_1:z:0?test_dataset_parallel_n20/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????³
,test_dataset_parallel_n20/dropout_2/IdentityIdentityCtest_dataset_parallel_n20/batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? ³
,test_dataset_parallel_n20/dropout_1/IdentityIdentityCtest_dataset_parallel_n20/batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????@°
*test_dataset_parallel_n20/dropout/IdentityIdentityAtest_dataset_parallel_n20/batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????z
8test_dataset_parallel_n20/max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :φ
4test_dataset_parallel_n20/max_pooling1d_2/ExpandDims
ExpandDims5test_dataset_parallel_n20/dropout_2/Identity:output:0Atest_dataset_parallel_n20/max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? θ
1test_dataset_parallel_n20/max_pooling1d_2/MaxPoolMaxPool=test_dataset_parallel_n20/max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????
 *
ksize
*
paddingVALID*
strides
Ε
1test_dataset_parallel_n20/max_pooling1d_2/SqueezeSqueeze:test_dataset_parallel_n20/max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????
 *
squeeze_dims
z
8test_dataset_parallel_n20/max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :φ
4test_dataset_parallel_n20/max_pooling1d_1/ExpandDims
ExpandDims5test_dataset_parallel_n20/dropout_1/Identity:output:0Atest_dataset_parallel_n20/max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@θ
1test_dataset_parallel_n20/max_pooling1d_1/MaxPoolMaxPool=test_dataset_parallel_n20/max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:?????????
@*
ksize
*
paddingVALID*
strides
Ε
1test_dataset_parallel_n20/max_pooling1d_1/SqueezeSqueeze:test_dataset_parallel_n20/max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????
@*
squeeze_dims
x
6test_dataset_parallel_n20/max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ρ
2test_dataset_parallel_n20/max_pooling1d/ExpandDims
ExpandDims3test_dataset_parallel_n20/dropout/Identity:output:0?test_dataset_parallel_n20/max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????ε
/test_dataset_parallel_n20/max_pooling1d/MaxPoolMaxPool;test_dataset_parallel_n20/max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
Β
/test_dataset_parallel_n20/max_pooling1d/SqueezeSqueeze8test_dataset_parallel_n20/max_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????
*
squeeze_dims
x
'test_dataset_parallel_n20/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   Σ
)test_dataset_parallel_n20/flatten/ReshapeReshape8test_dataset_parallel_n20/max_pooling1d/Squeeze:output:00test_dataset_parallel_n20/flatten/Const:output:0*
T0*(
_output_shapes
:?????????
z
)test_dataset_parallel_n20/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  Ω
+test_dataset_parallel_n20/flatten_1/ReshapeReshape:test_dataset_parallel_n20/max_pooling1d_1/Squeeze:output:02test_dataset_parallel_n20/flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????z
)test_dataset_parallel_n20/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  Ω
+test_dataset_parallel_n20/flatten_2/ReshapeReshape:test_dataset_parallel_n20/max_pooling1d_2/Squeeze:output:02test_dataset_parallel_n20/flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????ΐs
1test_dataset_parallel_n20/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Π
,test_dataset_parallel_n20/concatenate/concatConcatV22test_dataset_parallel_n20/flatten/Reshape:output:04test_dataset_parallel_n20/flatten_1/Reshape:output:04test_dataset_parallel_n20/flatten_2/Reshape:output:0:test_dataset_parallel_n20/concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????ΐ΅
5test_dataset_parallel_n20/dense/MatMul/ReadVariableOpReadVariableOp>test_dataset_parallel_n20_dense_matmul_readvariableop_resource*
_output_shapes
:	ΐ*
dtype0Ψ
&test_dataset_parallel_n20/dense/MatMulMatMul5test_dataset_parallel_n20/concatenate/concat:output:0=test_dataset_parallel_n20/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????²
6test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOpReadVariableOp?test_dataset_parallel_n20_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Φ
'test_dataset_parallel_n20/dense/BiasAddBiasAdd0test_dataset_parallel_n20/dense/MatMul:product:0>test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
IdentityIdentity0test_dataset_parallel_n20/dense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Σ
NoOpNoOpG^test_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOpI^test_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_1I^test_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_2K^test_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOpI^test_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOpK^test_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_1K^test_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_2M^test_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOpI^test_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOpK^test_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_1K^test_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_2M^test_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOp8^test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOpD^test_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOp:^test_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOpF^test_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:^test_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOpF^test_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp7^test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOp6^test_dataset_parallel_n20/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2
Ftest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOpFtest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp2
Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_1Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_12
Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_2Htest_dataset_parallel_n20/batch_normalization/batchnorm/ReadVariableOp_22
Jtest_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOpJtest_dataset_parallel_n20/batch_normalization/batchnorm/mul/ReadVariableOp2
Htest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOpHtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp2
Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_1Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_12
Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_2Jtest_dataset_parallel_n20/batch_normalization_1/batchnorm/ReadVariableOp_22
Ltest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOpLtest_dataset_parallel_n20/batch_normalization_1/batchnorm/mul/ReadVariableOp2
Htest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOpHtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp2
Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_1Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_12
Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_2Jtest_dataset_parallel_n20/batch_normalization_2/batchnorm/ReadVariableOp_22
Ltest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOpLtest_dataset_parallel_n20/batch_normalization_2/batchnorm/mul/ReadVariableOp2r
7test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOp7test_dataset_parallel_n20/conv1d/BiasAdd/ReadVariableOp2
Ctest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOpCtest_dataset_parallel_n20/conv1d/Conv1D/ExpandDims_1/ReadVariableOp2v
9test_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOp9test_dataset_parallel_n20/conv1d_1/BiasAdd/ReadVariableOp2
Etest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpEtest_dataset_parallel_n20/conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2v
9test_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOp9test_dataset_parallel_n20/conv1d_2/BiasAdd/ReadVariableOp2
Etest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpEtest_dataset_parallel_n20/conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2p
6test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOp6test_dataset_parallel_n20/dense/BiasAdd/ReadVariableOp2n
5test_dataset_parallel_n20/dense/MatMul/ReadVariableOp5test_dataset_parallel_n20/dense/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1
Ο
f
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs


c
D__inference_dropout_2_layer_call_and_return_conditional_losses_42478

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ͺ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:????????? s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:????????? m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:????????? ]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
λ

9__inference_test_dataset_parallel_n20_layer_call_fn_42707
input_1
unknown:
 
	unknown_0: 
	unknown_1:
@
	unknown_2:@ 
	unknown_3:

	unknown_4:	
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	ΐ

unknown_18:
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1

―
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43470

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Ζ

C__inference_conv1d_1_layer_call_and_return_conditional_losses_43319

inputsA
+conv1d_expanddims_1_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
@¬
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
§
ϋ
#__inference_signature_wrapper_42882
input_1
unknown:
 
	unknown_0: 
	unknown_1:
@
	unknown_2:@ 
	unknown_3:

	unknown_4:	
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	ΐ

unknown_18:
identity’StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_41867o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1
ͺ
E
)__inference_flatten_1_layer_call_fn_43720

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
Η	
ς
@__inference_dense_layer_call_and_return_conditional_losses_42326

inputs1
matmul_readvariableop_resource:	ΐ-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ΐ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????ΐ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs
°
e
+__inference_concatenate_layer_call_fn_43744
inputs_0
inputs_1
inputs_2
identityΝ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_42314a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????
:?????????:?????????ΐ:R N
(
_output_shapes
:?????????

"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/2
ΐ

%__inference_dense_layer_call_fn_43761

inputs
unknown:	ΐ
	unknown_0:
identity’StatefulPartitionedCallΨ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42326o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????ΐ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs
ξ

9__inference_test_dataset_parallel_n20_layer_call_fn_42927

inputs
unknown:
 
	unknown_0: 
	unknown_1:
@
	unknown_2:@ 
	unknown_3:

	unknown_4:	
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	ΐ

unknown_18:
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

±
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43390

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:??????????????????{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:??????????????????p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:??????????????????Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs

K
/__inference_max_pooling1d_1_layer_call_fn_43683

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
θ

9__inference_test_dataset_parallel_n20_layer_call_fn_42972

inputs
unknown:
 
	unknown_0: 
	unknown_1:
@
	unknown_2:@ 
	unknown_3:

	unknown_4:	
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	ΐ

unknown_18:
identity’StatefulPartitionedCallΩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
η
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_42263

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs

K
/__inference_max_pooling1d_2_layer_call_fn_43696

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ό
~
F__inference_concatenate_layer_call_and_return_conditional_losses_42314

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????ΐX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????
:?????????:?????????ΐ:P L
(
_output_shapes
:?????????

 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs
L
ψ	
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42829
input_1$
conv1d_2_42771:
 
conv1d_2_42773: $
conv1d_1_42776:
@
conv1d_1_42778:@#
conv1d_42781:

conv1d_42783:	)
batch_normalization_2_42786: )
batch_normalization_2_42788: )
batch_normalization_2_42790: )
batch_normalization_2_42792: )
batch_normalization_1_42795:@)
batch_normalization_1_42797:@)
batch_normalization_1_42799:@)
batch_normalization_1_42801:@(
batch_normalization_42804:	(
batch_normalization_42806:	(
batch_normalization_42808:	(
batch_normalization_42810:	
dense_42823:	ΐ
dense_42825:
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’dense/StatefulPartitionedCall’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’!dropout_2/StatefulPartitionedCallυ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_2_42771conv1d_2_42773*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181υ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_42776conv1d_1_42778*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203ξ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_42781conv1d_42783*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_42225
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_42786batch_normalization_2_42788batch_normalization_2_42790batch_normalization_2_42792*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42102
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_42795batch_normalization_1_42797batch_normalization_1_42799batch_normalization_1_42801*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_42020ϊ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_42804batch_normalization_42806batch_normalization_42808batch_normalization_42810*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41938
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42478€
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42455
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42432π
max_pooling1d_2/PartitionedCallPartitionedCall*dropout_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155π
max_pooling1d_1/PartitionedCallPartitionedCall*dropout_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140λ
max_pooling1d/PartitionedCallPartitionedCall(dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125Ω
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_42288ί
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296ί
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304₯
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_42314
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_42823dense_42825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ε
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1
Ν
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_43678

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_43611

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ρ

9__inference_test_dataset_parallel_n20_layer_call_fn_42376
input_1
unknown:
 
	unknown_0: 
	unknown_1:
@
	unknown_2:@ 
	unknown_3:

	unknown_4:	
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9:@

unknown_10:@

unknown_11:@

unknown_12:@

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	ΐ

unknown_18:
identity’StatefulPartitionedCallΰ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *]
fXRV
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42333o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1
ο
Ψ
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43089

inputsJ
4conv1d_2_conv1d_expanddims_1_readvariableop_resource:
 6
(conv1d_2_biasadd_readvariableop_resource: J
4conv1d_1_conv1d_expanddims_1_readvariableop_resource:
@6
(conv1d_1_biasadd_readvariableop_resource:@I
2conv1d_conv1d_expanddims_1_readvariableop_resource:
5
&conv1d_biasadd_readvariableop_resource:	E
7batch_normalization_2_batchnorm_readvariableop_resource: I
;batch_normalization_2_batchnorm_mul_readvariableop_resource: G
9batch_normalization_2_batchnorm_readvariableop_1_resource: G
9batch_normalization_2_batchnorm_readvariableop_2_resource: E
7batch_normalization_1_batchnorm_readvariableop_resource:@I
;batch_normalization_1_batchnorm_mul_readvariableop_resource:@G
9batch_normalization_1_batchnorm_readvariableop_1_resource:@G
9batch_normalization_1_batchnorm_readvariableop_2_resource:@D
5batch_normalization_batchnorm_readvariableop_resource:	H
9batch_normalization_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_batchnorm_readvariableop_1_resource:	F
7batch_normalization_batchnorm_readvariableop_2_resource:	7
$dense_matmul_readvariableop_resource:	ΐ3
%dense_biasadd_readvariableop_resource:
identity’,batch_normalization/batchnorm/ReadVariableOp’.batch_normalization/batchnorm/ReadVariableOp_1’.batch_normalization/batchnorm/ReadVariableOp_2’0batch_normalization/batchnorm/mul/ReadVariableOp’.batch_normalization_1/batchnorm/ReadVariableOp’0batch_normalization_1/batchnorm/ReadVariableOp_1’0batch_normalization_1/batchnorm/ReadVariableOp_2’2batch_normalization_1/batchnorm/mul/ReadVariableOp’.batch_normalization_2/batchnorm/ReadVariableOp’0batch_normalization_2/batchnorm/ReadVariableOp_1’0batch_normalization_2/batchnorm/ReadVariableOp_2’2batch_normalization_2/batchnorm/mul/ReadVariableOp’conv1d/BiasAdd/ReadVariableOp’)conv1d/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_1/BiasAdd/ReadVariableOp’+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp’conv1d_2/BiasAdd/ReadVariableOp’+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOpi
conv1d_2/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d_2/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_2/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
€
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
 *
dtype0b
 conv1d_2/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_2/Conv1D/ExpandDims_1
ExpandDims3conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
 Η
conv1d_2/Conv1DConv2D#conv1d_2/Conv1D/ExpandDims:output:0%conv1d_2/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides

conv1d_2/Conv1D/SqueezeSqueezeconv1d_2/Conv1D:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

ύ????????
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0
conv1d_2/BiasAddBiasAdd conv1d_2/Conv1D/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? f
conv1d_2/ReluReluconv1d_2/BiasAdd:output:0*
T0*+
_output_shapes
:????????? i
conv1d_1/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d_1/Conv1D/ExpandDims
ExpandDimsinputs'conv1d_1/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
€
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
@*
dtype0b
 conv1d_1/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : »
conv1d_1/Conv1D/ExpandDims_1
ExpandDims3conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
@Η
conv1d_1/Conv1DConv2D#conv1d_1/Conv1D/ExpandDims:output:0%conv1d_1/Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides

conv1d_1/Conv1D/SqueezeSqueezeconv1d_1/Conv1D:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

ύ????????
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv1d_1/BiasAddBiasAdd conv1d_1/Conv1D/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@f
conv1d_1/ReluReluconv1d_1/BiasAdd:output:0*
T0*+
_output_shapes
:?????????@g
conv1d/Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
conv1d/Conv1D/ExpandDims
ExpandDimsinputs%conv1d/Conv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????
‘
)conv1d/Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*#
_output_shapes
:
*
dtype0`
conv1d/Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : Ά
conv1d/Conv1D/ExpandDims_1
ExpandDims1conv1d/Conv1D/ExpandDims_1/ReadVariableOp:value:0'conv1d/Conv1D/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:
Β
conv1d/Conv1DConv2D!conv1d/Conv1D/ExpandDims:output:0#conv1d/Conv1D/ExpandDims_1:output:0*
T0*0
_output_shapes
:?????????*
paddingSAME*
strides

conv1d/Conv1D/SqueezeSqueezeconv1d/Conv1D:output:0*
T0*,
_output_shapes
:?????????*
squeeze_dims

ύ????????
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv1d/BiasAddBiasAddconv1d/Conv1D/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????c
conv1d/ReluReluconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:?????????’
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0j
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ή
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
: |
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
: ͺ
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0Ά
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: ¨
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Relu:activations:0'batch_normalization_2/batchnorm/mul:z:0*
T0*+
_output_shapes
:????????? ¦
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0΄
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
: ¦
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0΄
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
: Έ
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*+
_output_shapes
:????????? ’
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ή
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
:@|
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
:@ͺ
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0Ά
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@¨
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Relu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*+
_output_shapes
:?????????@¦
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0΄
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
:@¦
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0΄
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@Έ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*+
_output_shapes
:?????????@
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:΄
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:§
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0±
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:£
#batch_normalization/batchnorm/mul_1Mulconv1d/Relu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????£
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0―
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:£
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0―
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:³
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????
dropout_2/IdentityIdentity)batch_normalization_2/batchnorm/add_1:z:0*
T0*+
_output_shapes
:????????? 
dropout_1/IdentityIdentity)batch_normalization_1/batchnorm/add_1:z:0*
T0*+
_output_shapes
:?????????@|
dropout/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*,
_output_shapes
:?????????`
max_pooling1d_2/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¨
max_pooling1d_2/ExpandDims
ExpandDimsdropout_2/Identity:output:0'max_pooling1d_2/ExpandDims/dim:output:0*
T0*/
_output_shapes
:????????? ΄
max_pooling1d_2/MaxPoolMaxPool#max_pooling1d_2/ExpandDims:output:0*/
_output_shapes
:?????????
 *
ksize
*
paddingVALID*
strides

max_pooling1d_2/SqueezeSqueeze max_pooling1d_2/MaxPool:output:0*
T0*+
_output_shapes
:?????????
 *
squeeze_dims
`
max_pooling1d_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :¨
max_pooling1d_1/ExpandDims
ExpandDimsdropout_1/Identity:output:0'max_pooling1d_1/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????@΄
max_pooling1d_1/MaxPoolMaxPool#max_pooling1d_1/ExpandDims:output:0*/
_output_shapes
:?????????
@*
ksize
*
paddingVALID*
strides

max_pooling1d_1/SqueezeSqueeze max_pooling1d_1/MaxPool:output:0*
T0*+
_output_shapes
:?????????
@*
squeeze_dims
^
max_pooling1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :£
max_pooling1d/ExpandDims
ExpandDimsdropout/Identity:output:0%max_pooling1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:?????????±
max_pooling1d/MaxPoolMaxPool!max_pooling1d/ExpandDims:output:0*0
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides

max_pooling1d/SqueezeSqueezemax_pooling1d/MaxPool:output:0*
T0*,
_output_shapes
:?????????
*
squeeze_dims
^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   
flatten/ReshapeReshapemax_pooling1d/Squeeze:output:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????
`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  
flatten_1/ReshapeReshape max_pooling1d_1/Squeeze:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:?????????`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  
flatten_2/ReshapeReshape max_pooling1d_2/Squeeze:output:0flatten_2/Const:output:0*
T0*(
_output_shapes
:?????????ΐY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ξ
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????ΐ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	ΐ*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????e
IdentityIdentitydense/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????Λ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp/^batch_normalization_2/batchnorm/ReadVariableOp1^batch_normalization_2/batchnorm/ReadVariableOp_11^batch_normalization_2/batchnorm/ReadVariableOp_23^batch_normalization_2/batchnorm/mul/ReadVariableOp^conv1d/BiasAdd/ReadVariableOp*^conv1d/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_1/BiasAdd/ReadVariableOp,^conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp ^conv1d_2/BiasAdd/ReadVariableOp,^conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2`
.batch_normalization_2/batchnorm/ReadVariableOp.batch_normalization_2/batchnorm/ReadVariableOp2d
0batch_normalization_2/batchnorm/ReadVariableOp_10batch_normalization_2/batchnorm/ReadVariableOp_12d
0batch_normalization_2/batchnorm/ReadVariableOp_20batch_normalization_2/batchnorm/ReadVariableOp_22h
2batch_normalization_2/batchnorm/mul/ReadVariableOp2batch_normalization_2/batchnorm/mul/ReadVariableOp2>
conv1d/BiasAdd/ReadVariableOpconv1d/BiasAdd/ReadVariableOp2V
)conv1d/Conv1D/ExpandDims_1/ReadVariableOp)conv1d/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_1/BiasAdd/ReadVariableOpconv1d_1/BiasAdd/ReadVariableOp2Z
+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_1/Conv1D/ExpandDims_1/ReadVariableOp2B
conv1d_2/BiasAdd/ReadVariableOpconv1d_2/BiasAdd/ReadVariableOp2Z
+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp+conv1d_2/Conv1D/ExpandDims_1/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
&
λ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43424

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:??????????????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:??????????????????p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:??????????????????κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs

―
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42055

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :?????????????????? z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :?????????????????? o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
°
E
)__inference_dropout_1_layer_call_fn_43616

inputs
identityΆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42270d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ύ
`
D__inference_flatten_1_layer_call_and_return_conditional_losses_43726

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????
@:S O
+
_output_shapes
:?????????
@
 
_user_specified_nameinputs
έ
Π
5__inference_batch_normalization_2_layer_call_fn_43517

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42055|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs

b
)__inference_dropout_2_layer_call_fn_43648

inputs
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42478s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
ό%
ι
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42102

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :?????????????????? s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ’
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :?????????????????? h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :?????????????????? o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
θε
Γ$
!__inference__traced_restore_44146
file_prefix5
assignvariableop_conv1d_kernel:
-
assignvariableop_1_conv1d_bias:	8
"assignvariableop_2_conv1d_1_kernel:
@.
 assignvariableop_3_conv1d_1_bias:@8
"assignvariableop_4_conv1d_2_kernel:
 .
 assignvariableop_5_conv1d_2_bias: ;
,assignvariableop_6_batch_normalization_gamma:	:
+assignvariableop_7_batch_normalization_beta:	A
2assignvariableop_8_batch_normalization_moving_mean:	E
6assignvariableop_9_batch_normalization_moving_variance:	=
/assignvariableop_10_batch_normalization_1_gamma:@<
.assignvariableop_11_batch_normalization_1_beta:@C
5assignvariableop_12_batch_normalization_1_moving_mean:@G
9assignvariableop_13_batch_normalization_1_moving_variance:@=
/assignvariableop_14_batch_normalization_2_gamma: <
.assignvariableop_15_batch_normalization_2_beta: C
5assignvariableop_16_batch_normalization_2_moving_mean: G
9assignvariableop_17_batch_normalization_2_moving_variance: 3
 assignvariableop_18_dense_kernel:	ΐ,
assignvariableop_19_dense_bias:'
assignvariableop_20_adam_iter:	 )
assignvariableop_21_adam_beta_1: )
assignvariableop_22_adam_beta_2: (
assignvariableop_23_adam_decay: 0
&assignvariableop_24_adam_learning_rate: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: #
assignvariableop_27_total: #
assignvariableop_28_count: ?
(assignvariableop_29_adam_conv1d_kernel_m:
5
&assignvariableop_30_adam_conv1d_bias_m:	@
*assignvariableop_31_adam_conv1d_1_kernel_m:
@6
(assignvariableop_32_adam_conv1d_1_bias_m:@@
*assignvariableop_33_adam_conv1d_2_kernel_m:
 6
(assignvariableop_34_adam_conv1d_2_bias_m: C
4assignvariableop_35_adam_batch_normalization_gamma_m:	B
3assignvariableop_36_adam_batch_normalization_beta_m:	D
6assignvariableop_37_adam_batch_normalization_1_gamma_m:@C
5assignvariableop_38_adam_batch_normalization_1_beta_m:@D
6assignvariableop_39_adam_batch_normalization_2_gamma_m: C
5assignvariableop_40_adam_batch_normalization_2_beta_m: :
'assignvariableop_41_adam_dense_kernel_m:	ΐ3
%assignvariableop_42_adam_dense_bias_m:?
(assignvariableop_43_adam_conv1d_kernel_v:
5
&assignvariableop_44_adam_conv1d_bias_v:	@
*assignvariableop_45_adam_conv1d_1_kernel_v:
@6
(assignvariableop_46_adam_conv1d_1_bias_v:@@
*assignvariableop_47_adam_conv1d_2_kernel_v:
 6
(assignvariableop_48_adam_conv1d_2_bias_v: C
4assignvariableop_49_adam_batch_normalization_gamma_v:	B
3assignvariableop_50_adam_batch_normalization_beta_v:	D
6assignvariableop_51_adam_batch_normalization_1_gamma_v:@C
5assignvariableop_52_adam_batch_normalization_1_beta_v:@D
6assignvariableop_53_adam_batch_normalization_2_gamma_v: C
5assignvariableop_54_adam_batch_normalization_2_beta_v: :
'assignvariableop_55_adam_dense_kernel_v:	ΐ3
%assignvariableop_56_adam_dense_bias_v:
identity_58’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_47’AssignVariableOp_48’AssignVariableOp_49’AssignVariableOp_5’AssignVariableOp_50’AssignVariableOp_51’AssignVariableOp_52’AssignVariableOp_53’AssignVariableOp_54’AssignVariableOp_55’AssignVariableOp_56’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9Ρ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
::*
dtype0*χ
valueνBκ:B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHε
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
::*
dtype0*
value~B|:B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Γ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ώ
_output_shapesλ
θ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*H
dtypes>
<2:	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv1d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv1d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv1d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv1d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv1d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:‘
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_beta_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_decayIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_learning_rateIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_totalIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_countIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_conv1d_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_conv1d_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv1d_1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv1d_1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp*assignvariableop_33_adam_conv1d_2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv1d_2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_batch_normalization_gamma_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_batch_normalization_beta_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_37AssignVariableOp6assignvariableop_37_adam_batch_normalization_1_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_batch_normalization_1_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_batch_normalization_2_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_batch_normalization_2_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp%assignvariableop_42_adam_dense_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp(assignvariableop_43_adam_conv1d_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp&assignvariableop_44_adam_conv1d_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv1d_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv1d_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp*assignvariableop_47_adam_conv1d_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp(assignvariableop_48_adam_conv1d_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:₯
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_batch_normalization_gamma_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:€
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_batch_normalization_beta_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_51AssignVariableOp6assignvariableop_51_adam_batch_normalization_1_gamma_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_batch_normalization_1_beta_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_2_gamma_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_2_beta_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_dense_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp%assignvariableop_56_adam_dense_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ΅

Identity_57Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_58IdentityIdentity_57:output:0^NoOp_1*
T0*
_output_shapes
: ’

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_58Identity_58:output:0*
_input_shapesv
t: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ό%
ι
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43504

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :??????????????????@s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ’
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :??????????????????@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :??????????????????@o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Ζ

C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181

inputsA
+conv1d_expanddims_1_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
 *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
 ¬
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

`
'__inference_dropout_layer_call_fn_43594

inputs
identity’StatefulPartitionedCallΕ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42432t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
Η

F__inference_concatenate_layer_call_and_return_conditional_losses_43752
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*(
_output_shapes
:?????????ΐX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:?????????ΐ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:?????????
:?????????:?????????ΐ:R N
(
_output_shapes
:?????????

"
_user_specified_name
inputs/0:RN
(
_output_shapes
:?????????
"
_user_specified_name
inputs/1:RN
(
_output_shapes
:?????????ΐ
"
_user_specified_name
inputs/2
&
λ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41938

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(i
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:??????????????????s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       £
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:*
	keep_dims(o
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 u
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:q
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:??????????????????i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:??????????????????p
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*5
_output_shapes#
!:??????????????????κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
Ϋ
Π
5__inference_batch_normalization_1_layer_call_fn_43450

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_42020|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
Ζ

C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203

inputsA
+conv1d_expanddims_1_readvariableop_resource:
@-
biasadd_readvariableop_resource:@
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
@*
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
@¬
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:?????????@*
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????@T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????@e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:?????????@
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs

―
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43550

inputs/
!batchnorm_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: 1
#batchnorm_readvariableop_1_resource: 1
#batchnorm_readvariableop_2_resource: 
identity’batchnorm/ReadVariableOp’batchnorm/ReadVariableOp_1’batchnorm/ReadVariableOp_2’batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :?????????????????? z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :?????????????????? o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? Ί
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs

b
)__inference_dropout_1_layer_call_fn_43621

inputs
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42455s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
°
E
)__inference_dropout_2_layer_call_fn_43643

inputs
identityΆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42263d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs
Ϋ
Π
5__inference_batch_normalization_2_layer_call_fn_43530

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42102|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs

I
-__inference_max_pooling1d_layer_call_fn_43670

inputs
identityΜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *=
_output_shapes+
):'???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125v
IdentityIdentityPartitionedCall:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
°
C
'__inference_dropout_layer_call_fn_43589

inputs
identity΅
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42277e
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ί
?
3__inference_batch_normalization_layer_call_fn_43370

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41938}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs


c
D__inference_dropout_1_layer_call_and_return_conditional_losses_42455

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?h
dropout/MulMulinputsdropout/Const:output:0*
T0*+
_output_shapes
:?????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:?????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>ͺ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:?????????@s
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:?????????@m
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:?????????@]
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ύ
^
B__inference_flatten_layer_call_and_return_conditional_losses_42288

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????
Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????
:T P
,
_output_shapes
:?????????

 
_user_specified_nameinputs
η
b
D__inference_dropout_2_layer_call_and_return_conditional_losses_43653

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:????????? _

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:????????? :S O
+
_output_shapes
:????????? 
 
_user_specified_nameinputs


a
B__inference_dropout_layer_call_and_return_conditional_losses_42432

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?i
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:?????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:?????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:?????????t
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:?????????n
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:?????????^
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????:T P
,
_output_shapes
:?????????
 
_user_specified_nameinputs
ό%
ι
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43584

inputs5
'assignmovingavg_readvariableop_resource: 7
)assignmovingavg_1_readvariableop_resource: 3
%batchnorm_mul_readvariableop_resource: /
!batchnorm_readvariableop_resource: 
identity’AssignMovingAvg’AssignMovingAvg/ReadVariableOp’AssignMovingAvg_1’ AssignMovingAvg_1/ReadVariableOp’batchnorm/ReadVariableOp’batchnorm/mul/ReadVariableOpo
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(h
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :?????????????????? s
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       ’
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
: x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
: ¬
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
Χ#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
: *
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
: ~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
: ΄
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: ~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: p
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :?????????????????? h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :?????????????????? o
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? κ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:?????????????????? : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs
Η	
ς
@__inference_dense_layer_call_and_return_conditional_losses_43771

inputs1
matmul_readvariableop_resource:	ΐ-
biasadd_readvariableop_resource:
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ΐ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????ΐ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????ΐ
 
_user_specified_nameinputs
ΨG
	
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42768
input_1$
conv1d_2_42710:
 
conv1d_2_42712: $
conv1d_1_42715:
@
conv1d_1_42717:@#
conv1d_42720:

conv1d_42722:	)
batch_normalization_2_42725: )
batch_normalization_2_42727: )
batch_normalization_2_42729: )
batch_normalization_2_42731: )
batch_normalization_1_42734:@)
batch_normalization_1_42736:@)
batch_normalization_1_42738:@)
batch_normalization_1_42740:@(
batch_normalization_42743:	(
batch_normalization_42745:	(
batch_normalization_42747:	(
batch_normalization_42749:	
dense_42762:	ΐ
dense_42764:
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’conv1d/StatefulPartitionedCall’ conv1d_1/StatefulPartitionedCall’ conv1d_2/StatefulPartitionedCall’dense/StatefulPartitionedCallυ
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_2_42710conv1d_2_42712*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181υ
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_1_42715conv1d_1_42717*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_42203ξ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_42720conv1d_42722*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_42225
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_42725batch_normalization_2_42727batch_normalization_2_42729batch_normalization_2_42731*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_42055
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_42734batch_normalization_1_42736batch_normalization_1_42738batch_normalization_1_42740*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41973ό
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_42743batch_normalization_42745batch_normalization_42747batch_normalization_42749*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41891π
dropout_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_2_layer_call_and_return_conditional_losses_42263π
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_42270λ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_42277θ
max_pooling1d_2/PartitionedCallPartitionedCall"dropout_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_42155θ
max_pooling1d_1/PartitionedCallPartitionedCall"dropout_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????
@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_42140γ
max_pooling1d/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125Ω
flatten/PartitionedCallPartitionedCall&max_pooling1d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_42288ί
flatten_1/PartitionedCallPartitionedCall(max_pooling1d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_1_layer_call_and_return_conditional_losses_42296ί
flatten_2/PartitionedCallPartitionedCall(max_pooling1d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_42304₯
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????ΐ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_42314
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_42762dense_42764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_42326u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????Ϋ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????
: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
+
_output_shapes
:?????????

!
_user_specified_name	input_1
Χ

(__inference_conv1d_2_layer_call_fn_43328

inputs
unknown:
 
	unknown_0: 
identity’StatefulPartitionedCallί
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_42181s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
Ν
d
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_42125

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
α
?
3__inference_batch_normalization_layer_call_fn_43357

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:??????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_41891}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:??????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):??????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:??????????????????
 
_user_specified_nameinputs
¨
C
'__inference_flatten_layer_call_fn_43709

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_42288a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????
:T P
,
_output_shapes
:?????????

 
_user_specified_nameinputs
έ
Π
5__inference_batch_normalization_1_layer_call_fn_43437

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_41973|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs
η
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_43626

inputs

identity_1R
IdentityIdentityinputs*
T0*+
_output_shapes
:?????????@_

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:?????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@:S O
+
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ο
f
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_43691

inputs
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*A
_output_shapes/
-:+???????????????????????????¦
MaxPoolMaxPoolExpandDims:output:0*A
_output_shapes/
-:+???????????????????????????*
ksize
*
paddingVALID*
strides

SqueezeSqueezeMaxPool:output:0*
T0*=
_output_shapes+
):'???????????????????????????*
squeeze_dims
n
IdentityIdentitySqueeze:output:0*
T0*=
_output_shapes+
):'???????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'???????????????????????????:e a
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
Ζ

C__inference_conv1d_2_layer_call_and_return_conditional_losses_43344

inputsA
+conv1d_expanddims_1_readvariableop_resource:
 -
biasadd_readvariableop_resource: 
identity’BiasAdd/ReadVariableOp’"Conv1D/ExpandDims_1/ReadVariableOp`
Conv1D/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ύ????????
Conv1D/ExpandDims
ExpandDimsinputsConv1D/ExpandDims/dim:output:0*
T0*/
_output_shapes
:?????????

"Conv1D/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:
 *
dtype0Y
Conv1D/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :  
Conv1D/ExpandDims_1
ExpandDims*Conv1D/ExpandDims_1/ReadVariableOp:value:0 Conv1D/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:
 ¬
Conv1DConv2DConv1D/ExpandDims:output:0Conv1D/ExpandDims_1:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides

Conv1D/SqueezeSqueezeConv1D:output:0*
T0*+
_output_shapes
:????????? *
squeeze_dims

ύ????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv1D/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:????????? T
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:????????? e
IdentityIdentityRelu:activations:0^NoOp*
T0*+
_output_shapes
:????????? 
NoOpNoOp^BiasAdd/ReadVariableOp#^Conv1D/ExpandDims_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"Conv1D/ExpandDims_1/ReadVariableOp"Conv1D/ExpandDims_1/ReadVariableOp:S O
+
_output_shapes
:?????????

 
_user_specified_nameinputs
Ύ
^
B__inference_flatten_layer_call_and_return_conditional_losses_43715

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????
Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????
"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????
:T P
,
_output_shapes
:?????????

 
_user_specified_nameinputs"ΏL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
?
input_14
serving_default_input_1:0?????????
9
dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:»¨

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer_with_weights-6
layer-17
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
έ
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias
 $_jit_compiled_convolution_op"
_tf_keras_layer
έ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias
 -_jit_compiled_convolution_op"
_tf_keras_layer
έ
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses

4kernel
5bias
 6_jit_compiled_convolution_op"
_tf_keras_layer
κ
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
κ
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
κ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance"
_tf_keras_layer
Ό
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
\__call__
*]&call_and_return_all_conditional_losses
^_random_generator"
_tf_keras_layer
Ό
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses
e_random_generator"
_tf_keras_layer
Ό
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses
l_random_generator"
_tf_keras_layer
₯
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses"
_tf_keras_layer
₯
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
ͺ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Γ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Έ
"0
#1
+2
,3
44
55
>6
?7
@8
A9
I10
J11
K12
L13
T14
U15
V16
W17
18
19"
trackable_list_wrapper

"0
#1
+2
,3
44
55
>6
?7
I8
J9
T10
U11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
Ο
non_trainable_variables
 layers
‘metrics
 ’layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
’
€trace_0
₯trace_1
¦trace_2
§trace_32―
9__inference_test_dataset_parallel_n20_layer_call_fn_42376
9__inference_test_dataset_parallel_n20_layer_call_fn_42927
9__inference_test_dataset_parallel_n20_layer_call_fn_42972
9__inference_test_dataset_parallel_n20_layer_call_fn_42707ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z€trace_0z₯trace_1z¦trace_2z§trace_3

¨trace_0
©trace_1
ͺtrace_2
«trace_32
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43089
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43269
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42768
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42829ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 z¨trace_0z©trace_1zͺtrace_2z«trace_3
ΛBΘ
 __inference__wrapped_model_41867input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τ
	¬iter
­beta_1
?beta_2

―decay
°learning_rate"mΐ#mΑ+mΒ,mΓ4mΔ5mΕ>mΖ?mΗImΘJmΙTmΚUmΛ	mΜ	mΝ"vΞ#vΟ+vΠ,vΡ4v?5vΣ>vΤ?vΥIvΦJvΧTvΨUvΩ	vΪ	vΫ"
	optimizer
-
±serving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
²non_trainable_variables
³layers
΄metrics
 ΅layer_regularization_losses
Άlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
μ
·trace_02Ν
&__inference_conv1d_layer_call_fn_43278’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z·trace_0

Έtrace_02θ
A__inference_conv1d_layer_call_and_return_conditional_losses_43294’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΈtrace_0
$:"
2conv1d/kernel
:2conv1d/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ήnon_trainable_variables
Ίlayers
»metrics
 Όlayer_regularization_losses
½layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
ξ
Ύtrace_02Ο
(__inference_conv1d_1_layer_call_fn_43303’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΎtrace_0

Ώtrace_02κ
C__inference_conv1d_1_layer_call_and_return_conditional_losses_43319’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΏtrace_0
%:#
@2conv1d_1/kernel
:@2conv1d_1/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ΐnon_trainable_variables
Αlayers
Βmetrics
 Γlayer_regularization_losses
Δlayer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
ξ
Εtrace_02Ο
(__inference_conv1d_2_layer_call_fn_43328’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΕtrace_0

Ζtrace_02κ
C__inference_conv1d_2_layer_call_and_return_conditional_losses_43344’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 zΖtrace_0
%:#
 2conv1d_2/kernel
: 2conv1d_2/bias
΄2±?
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 0
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ηnon_trainable_variables
Θlayers
Ιmetrics
 Κlayer_regularization_losses
Λlayer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
ά
Μtrace_0
Νtrace_12‘
3__inference_batch_normalization_layer_call_fn_43357
3__inference_batch_normalization_layer_call_fn_43370΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΜtrace_0zΝtrace_1

Ξtrace_0
Οtrace_12Χ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43390
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43424΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΞtrace_0zΟtrace_1
 "
trackable_list_wrapper
(:&2batch_normalization/gamma
':%2batch_normalization/beta
0:. (2batch_normalization/moving_mean
4:2 (2#batch_normalization/moving_variance
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Πnon_trainable_variables
Ρlayers
?metrics
 Σlayer_regularization_losses
Τlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ΰ
Υtrace_0
Φtrace_12₯
5__inference_batch_normalization_1_layer_call_fn_43437
5__inference_batch_normalization_1_layer_call_fn_43450΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΥtrace_0zΦtrace_1

Χtrace_0
Ψtrace_12Ϋ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43470
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43504΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΧtrace_0zΨtrace_1
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
T0
U1
V2
W3"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ωnon_trainable_variables
Ϊlayers
Ϋmetrics
 άlayer_regularization_losses
έlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ΰ
ήtrace_0
ίtrace_12₯
5__inference_batch_normalization_2_layer_call_fn_43517
5__inference_batch_normalization_2_layer_call_fn_43530΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zήtrace_0zίtrace_1

ΰtrace_0
αtrace_12Ϋ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43550
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43584΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zΰtrace_0zαtrace_1
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
βnon_trainable_variables
γlayers
δmetrics
 εlayer_regularization_losses
ζlayer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
\__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
Δ
ηtrace_0
θtrace_12
'__inference_dropout_layer_call_fn_43589
'__inference_dropout_layer_call_fn_43594΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zηtrace_0zθtrace_1
ϊ
ιtrace_0
κtrace_12Ώ
B__inference_dropout_layer_call_and_return_conditional_losses_43599
B__inference_dropout_layer_call_and_return_conditional_losses_43611΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zιtrace_0zκtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
λnon_trainable_variables
μlayers
νmetrics
 ξlayer_regularization_losses
οlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
Θ
πtrace_0
ρtrace_12
)__inference_dropout_1_layer_call_fn_43616
)__inference_dropout_1_layer_call_fn_43621΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zπtrace_0zρtrace_1
ώ
ςtrace_0
σtrace_12Γ
D__inference_dropout_1_layer_call_and_return_conditional_losses_43626
D__inference_dropout_1_layer_call_and_return_conditional_losses_43638΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zςtrace_0zσtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
τnon_trainable_variables
υlayers
φmetrics
 χlayer_regularization_losses
ψlayer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
Θ
ωtrace_0
ϊtrace_12
)__inference_dropout_2_layer_call_fn_43643
)__inference_dropout_2_layer_call_fn_43648΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zωtrace_0zϊtrace_1
ώ
ϋtrace_0
όtrace_12Γ
D__inference_dropout_2_layer_call_and_return_conditional_losses_43653
D__inference_dropout_2_layer_call_and_return_conditional_losses_43665΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 zϋtrace_0zόtrace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ύnon_trainable_variables
ώlayers
?metrics
 layer_regularization_losses
layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
σ
trace_02Τ
-__inference_max_pooling1d_layer_call_fn_43670’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ο
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_43678’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
υ
trace_02Φ
/__inference_max_pooling1d_1_layer_call_fn_43683’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ρ
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_43691’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
υ
trace_02Φ
/__inference_max_pooling1d_2_layer_call_fn_43696’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ρ
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_43704’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
·
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ν
trace_02Ξ
'__inference_flatten_layer_call_fn_43709’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02ι
B__inference_flatten_layer_call_and_return_conditional_losses_43715’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ο
trace_02Π
)__inference_flatten_1_layer_call_fn_43720’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0

trace_02λ
D__inference_flatten_1_layer_call_and_return_conditional_losses_43726’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
 non_trainable_variables
‘layers
’metrics
 £layer_regularization_losses
€layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ο
₯trace_02Π
)__inference_flatten_2_layer_call_fn_43731’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z₯trace_0

¦trace_02λ
D__inference_flatten_2_layer_call_and_return_conditional_losses_43737’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z¦trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
§non_trainable_variables
¨layers
©metrics
 ͺlayer_regularization_losses
«layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ρ
¬trace_02?
+__inference_concatenate_layer_call_fn_43744’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z¬trace_0

­trace_02ν
F__inference_concatenate_layer_call_and_return_conditional_losses_43752’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z­trace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?non_trainable_variables
―layers
°metrics
 ±layer_regularization_losses
²layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
λ
³trace_02Μ
%__inference_dense_layer_call_fn_43761’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z³trace_0

΄trace_02η
@__inference_dense_layer_call_and_return_conditional_losses_43771’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 z΄trace_0
:	ΐ2dense/kernel
:2
dense/bias
J
@0
A1
K2
L3
V4
W5"
trackable_list_wrapper
¦
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
0
΅0
Ά1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
9__inference_test_dataset_parallel_n20_layer_call_fn_42376input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
9__inference_test_dataset_parallel_n20_layer_call_fn_42927inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
9__inference_test_dataset_parallel_n20_layer_call_fn_42972inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
9__inference_test_dataset_parallel_n20_layer_call_fn_42707input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¦B£
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43089inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
¦B£
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43269inputs"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
§B€
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42768input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
§B€
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42829input_1"ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ΚBΗ
#__inference_signature_wrapper_42882input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΪBΧ
&__inference_conv1d_layer_call_fn_43278inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
υBς
A__inference_conv1d_layer_call_and_return_conditional_losses_43294inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
άBΩ
(__inference_conv1d_1_layer_call_fn_43303inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_1_layer_call_and_return_conditional_losses_43319inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
άBΩ
(__inference_conv1d_2_layer_call_fn_43328inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
χBτ
C__inference_conv1d_2_layer_call_and_return_conditional_losses_43344inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ωBφ
3__inference_batch_normalization_layer_call_fn_43357inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ωBφ
3__inference_batch_normalization_layer_call_fn_43370inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43390inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43424inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ϋBψ
5__inference_batch_normalization_1_layer_call_fn_43437inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ϋBψ
5__inference_batch_normalization_1_layer_call_fn_43450inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43470inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43504inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ϋBψ
5__inference_batch_normalization_2_layer_call_fn_43517inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ϋBψ
5__inference_batch_normalization_2_layer_call_fn_43530inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43550inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43584inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
νBκ
'__inference_dropout_layer_call_fn_43589inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
νBκ
'__inference_dropout_layer_call_fn_43594inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_43599inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
B__inference_dropout_layer_call_and_return_conditional_losses_43611inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
οBμ
)__inference_dropout_1_layer_call_fn_43616inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_1_layer_call_fn_43621inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_43626inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_1_layer_call_and_return_conditional_losses_43638inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
οBμ
)__inference_dropout_2_layer_call_fn_43643inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
οBμ
)__inference_dropout_2_layer_call_fn_43648inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_43653inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
B
D__inference_dropout_2_layer_call_and_return_conditional_losses_43665inputs"΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
αBή
-__inference_max_pooling1d_layer_call_fn_43670inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
όBω
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_43678inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
γBΰ
/__inference_max_pooling1d_1_layer_call_fn_43683inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ώBϋ
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_43691inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
γBΰ
/__inference_max_pooling1d_2_layer_call_fn_43696inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ώBϋ
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_43704inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΫBΨ
'__inference_flatten_layer_call_fn_43709inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
φBσ
B__inference_flatten_layer_call_and_return_conditional_losses_43715inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_flatten_1_layer_call_fn_43720inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_flatten_1_layer_call_and_return_conditional_losses_43726inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
έBΪ
)__inference_flatten_2_layer_call_fn_43731inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ψBυ
D__inference_flatten_2_layer_call_and_return_conditional_losses_43737inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
υBς
+__inference_concatenate_layer_call_fn_43744inputs/0inputs/1inputs/2"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
B
F__inference_concatenate_layer_call_and_return_conditional_losses_43752inputs/0inputs/1inputs/2"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΩBΦ
%__inference_dense_layer_call_fn_43761inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
τBρ
@__inference_dense_layer_call_and_return_conditional_losses_43771inputs"’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
R
·	variables
Έ	keras_api

Ήtotal

Ίcount"
_tf_keras_metric
c
»	variables
Ό	keras_api

½total

Ύcount
Ώ
_fn_kwargs"
_tf_keras_metric
0
Ή0
Ί1"
trackable_list_wrapper
.
·	variables"
_generic_user_object
:  (2total
:  (2count
0
½0
Ύ1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
):'
2Adam/conv1d/kernel/m
:2Adam/conv1d/bias/m
*:(
@2Adam/conv1d_1/kernel/m
 :@2Adam/conv1d_1/bias/m
*:(
 2Adam/conv1d_2/kernel/m
 : 2Adam/conv1d_2/bias/m
-:+2 Adam/batch_normalization/gamma/m
,:*2Adam/batch_normalization/beta/m
.:,@2"Adam/batch_normalization_1/gamma/m
-:+@2!Adam/batch_normalization_1/beta/m
.:, 2"Adam/batch_normalization_2/gamma/m
-:+ 2!Adam/batch_normalization_2/beta/m
$:"	ΐ2Adam/dense/kernel/m
:2Adam/dense/bias/m
):'
2Adam/conv1d/kernel/v
:2Adam/conv1d/bias/v
*:(
@2Adam/conv1d_1/kernel/v
 :@2Adam/conv1d_1/bias/v
*:(
 2Adam/conv1d_2/kernel/v
 : 2Adam/conv1d_2/bias/v
-:+2 Adam/batch_normalization/gamma/v
,:*2Adam/batch_normalization/beta/v
.:,@2"Adam/batch_normalization_1/gamma/v
-:+@2!Adam/batch_normalization_1/beta/v
.:, 2"Adam/batch_normalization_2/gamma/v
-:+ 2!Adam/batch_normalization_2/beta/v
$:"	ΐ2Adam/dense/kernel/v
:2Adam/dense/bias/v‘
 __inference__wrapped_model_41867}45+,"#WTVULIKJA>@?4’1
*’'
%"
input_1?????????

ͺ "-ͺ*
(
dense
dense?????????Π
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43470|LIKJ@’=
6’3
-*
inputs??????????????????@
p 
ͺ "2’/
(%
0??????????????????@
 Π
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_43504|KLIJ@’=
6’3
-*
inputs??????????????????@
p
ͺ "2’/
(%
0??????????????????@
 ¨
5__inference_batch_normalization_1_layer_call_fn_43437oLIKJ@’=
6’3
-*
inputs??????????????????@
p 
ͺ "%"??????????????????@¨
5__inference_batch_normalization_1_layer_call_fn_43450oKLIJ@’=
6’3
-*
inputs??????????????????@
p
ͺ "%"??????????????????@Π
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43550|WTVU@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "2’/
(%
0?????????????????? 
 Π
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_43584|VWTU@’=
6’3
-*
inputs?????????????????? 
p
ͺ "2’/
(%
0?????????????????? 
 ¨
5__inference_batch_normalization_2_layer_call_fn_43517oWTVU@’=
6’3
-*
inputs?????????????????? 
p 
ͺ "%"?????????????????? ¨
5__inference_batch_normalization_2_layer_call_fn_43530oVWTU@’=
6’3
-*
inputs?????????????????? 
p
ͺ "%"?????????????????? Π
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43390~A>@?A’>
7’4
.+
inputs??????????????????
p 
ͺ "3’0
)&
0??????????????????
 Π
N__inference_batch_normalization_layer_call_and_return_conditional_losses_43424~@A>?A’>
7’4
.+
inputs??????????????????
p
ͺ "3’0
)&
0??????????????????
 ¨
3__inference_batch_normalization_layer_call_fn_43357qA>@?A’>
7’4
.+
inputs??????????????????
p 
ͺ "&#??????????????????¨
3__inference_batch_normalization_layer_call_fn_43370q@A>?A’>
7’4
.+
inputs??????????????????
p
ͺ "&#??????????????????χ
F__inference_concatenate_layer_call_and_return_conditional_losses_43752¬’~
w’t
ro
# 
inputs/0?????????

# 
inputs/1?????????
# 
inputs/2?????????ΐ
ͺ "&’#

0?????????ΐ
 Ο
+__inference_concatenate_layer_call_fn_43744’~
w’t
ro
# 
inputs/0?????????

# 
inputs/1?????????
# 
inputs/2?????????ΐ
ͺ "?????????ΐ«
C__inference_conv1d_1_layer_call_and_return_conditional_losses_43319d+,3’0
)’&
$!
inputs?????????

ͺ ")’&

0?????????@
 
(__inference_conv1d_1_layer_call_fn_43303W+,3’0
)’&
$!
inputs?????????

ͺ "?????????@«
C__inference_conv1d_2_layer_call_and_return_conditional_losses_43344d453’0
)’&
$!
inputs?????????

ͺ ")’&

0????????? 
 
(__inference_conv1d_2_layer_call_fn_43328W453’0
)’&
$!
inputs?????????

ͺ "????????? ͺ
A__inference_conv1d_layer_call_and_return_conditional_losses_43294e"#3’0
)’&
$!
inputs?????????

ͺ "*’'
 
0?????????
 
&__inference_conv1d_layer_call_fn_43278X"#3’0
)’&
$!
inputs?????????

ͺ "?????????£
@__inference_dense_layer_call_and_return_conditional_losses_43771_0’-
&’#
!
inputs?????????ΐ
ͺ "%’"

0?????????
 {
%__inference_dense_layer_call_fn_43761R0’-
&’#
!
inputs?????????ΐ
ͺ "?????????¬
D__inference_dropout_1_layer_call_and_return_conditional_losses_43626d7’4
-’*
$!
inputs?????????@
p 
ͺ ")’&

0?????????@
 ¬
D__inference_dropout_1_layer_call_and_return_conditional_losses_43638d7’4
-’*
$!
inputs?????????@
p
ͺ ")’&

0?????????@
 
)__inference_dropout_1_layer_call_fn_43616W7’4
-’*
$!
inputs?????????@
p 
ͺ "?????????@
)__inference_dropout_1_layer_call_fn_43621W7’4
-’*
$!
inputs?????????@
p
ͺ "?????????@¬
D__inference_dropout_2_layer_call_and_return_conditional_losses_43653d7’4
-’*
$!
inputs????????? 
p 
ͺ ")’&

0????????? 
 ¬
D__inference_dropout_2_layer_call_and_return_conditional_losses_43665d7’4
-’*
$!
inputs????????? 
p
ͺ ")’&

0????????? 
 
)__inference_dropout_2_layer_call_fn_43643W7’4
-’*
$!
inputs????????? 
p 
ͺ "????????? 
)__inference_dropout_2_layer_call_fn_43648W7’4
-’*
$!
inputs????????? 
p
ͺ "????????? ¬
B__inference_dropout_layer_call_and_return_conditional_losses_43599f8’5
.’+
%"
inputs?????????
p 
ͺ "*’'
 
0?????????
 ¬
B__inference_dropout_layer_call_and_return_conditional_losses_43611f8’5
.’+
%"
inputs?????????
p
ͺ "*’'
 
0?????????
 
'__inference_dropout_layer_call_fn_43589Y8’5
.’+
%"
inputs?????????
p 
ͺ "?????????
'__inference_dropout_layer_call_fn_43594Y8’5
.’+
%"
inputs?????????
p
ͺ "?????????₯
D__inference_flatten_1_layer_call_and_return_conditional_losses_43726]3’0
)’&
$!
inputs?????????
@
ͺ "&’#

0?????????
 }
)__inference_flatten_1_layer_call_fn_43720P3’0
)’&
$!
inputs?????????
@
ͺ "?????????₯
D__inference_flatten_2_layer_call_and_return_conditional_losses_43737]3’0
)’&
$!
inputs?????????
 
ͺ "&’#

0?????????ΐ
 }
)__inference_flatten_2_layer_call_fn_43731P3’0
)’&
$!
inputs?????????
 
ͺ "?????????ΐ€
B__inference_flatten_layer_call_and_return_conditional_losses_43715^4’1
*’'
%"
inputs?????????

ͺ "&’#

0?????????

 |
'__inference_flatten_layer_call_fn_43709Q4’1
*’'
%"
inputs?????????

ͺ "?????????
Σ
J__inference_max_pooling1d_1_layer_call_and_return_conditional_losses_43691E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 ͺ
/__inference_max_pooling1d_1_layer_call_fn_43683wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????Σ
J__inference_max_pooling1d_2_layer_call_and_return_conditional_losses_43704E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 ͺ
/__inference_max_pooling1d_2_layer_call_fn_43696wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????Ρ
H__inference_max_pooling1d_layer_call_and_return_conditional_losses_43678E’B
;’8
63
inputs'???????????????????????????
ͺ ";’8
1.
0'???????????????????????????
 ¨
-__inference_max_pooling1d_layer_call_fn_43670wE’B
;’8
63
inputs'???????????????????????????
ͺ ".+'???????????????????????????°
#__inference_signature_wrapper_4288245+,"#WTVULIKJA>@??’<
’ 
5ͺ2
0
input_1%"
input_1?????????
"-ͺ*
(
dense
dense?????????Υ
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42768}45+,"#WTVULIKJA>@?<’9
2’/
%"
input_1?????????

p 

 
ͺ "%’"

0?????????
 Υ
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_42829}45+,"#VWTUKLIJ@A>?<’9
2’/
%"
input_1?????????

p

 
ͺ "%’"

0?????????
 Τ
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43089|45+,"#WTVULIKJA>@?;’8
1’.
$!
inputs?????????

p 

 
ͺ "%’"

0?????????
 Τ
T__inference_test_dataset_parallel_n20_layer_call_and_return_conditional_losses_43269|45+,"#VWTUKLIJ@A>?;’8
1’.
$!
inputs?????????

p

 
ͺ "%’"

0?????????
 ­
9__inference_test_dataset_parallel_n20_layer_call_fn_42376p45+,"#WTVULIKJA>@?<’9
2’/
%"
input_1?????????

p 

 
ͺ "?????????­
9__inference_test_dataset_parallel_n20_layer_call_fn_42707p45+,"#VWTUKLIJ@A>?<’9
2’/
%"
input_1?????????

p

 
ͺ "?????????¬
9__inference_test_dataset_parallel_n20_layer_call_fn_42927o45+,"#WTVULIKJA>@?;’8
1’.
$!
inputs?????????

p 

 
ͺ "?????????¬
9__inference_test_dataset_parallel_n20_layer_call_fn_42972o45+,"#VWTUKLIJ@A>?;’8
1’.
$!
inputs?????????

p

 
ͺ "?????????