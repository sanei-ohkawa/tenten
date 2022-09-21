"""
トーチの勉強

tensorflowと共存するやつ
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
CUDA 11.2 cuDNN8.1
"""

import torch
import numpy as np

### テンソルについて

# 型は小数がfloat32, 整数がint64

# 直接作る
tensor = torch.tensor([[1, 2], [3, 4]])

# numpyからテンソルにする
# メモリを共有してるみたいなので、元のnumpyを変えたらtensorも変わる
ndarray = np.array([[1, 2], [3, 4]])
tensor = torch.from_numpy(ndarray)

# シェイプ指定で作る
rand_tensor = torch.rand([2, 2])  # 0～1
ones_tensor = torch.ones([2, 2])
zeros_tensor = torch.zeros([2, 2])

# numpyにする
ndarray = tensor.numpy()

# スカラーの値をpythonのfloatにする
tensor = torch.tensor([1])
value = tensor.item()

# 属性取得
tensor = torch.rand([2, 2])
print(tensor.shape)
print(tensor.dtype)

# スライスの代入が可能
tensor[:, 1] = 0

# 結合
tensor = torch.cat([tensor, tensor, tensor], dim=1)

# リシェイプ
tensor = tensor.reshape([3, 4])

# 軸の交換
tensor = tensor.permute(1, 0)

# 行列積 これ全部同じ
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# 要素の掛け算 これ全部同じ
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# インプレース
# 演算履歴が無くなって微分できなくなるからネットワークの中では気軽に使えない
# 以下代表
abs_() absolute_() add_() div_() exp_() mul_() clip_()
tensor[:, 1] = 0  # スライスの代入
tensor += 2.0


### GPUとCPUの操作

# GPUが使えるか確認
print(torch.cuda.is_available())

# テンソルをGPUに移動
# GPUにあるテンソルとCPUにあるテンソルで操作を行うとエラー
inputs = inputs.to("cuda")

# テンソルをCPUに移動
outputs = outputs.to("cpu")

# テンソルがgpuにいるかcpuにいるか
print(tensor.device)

# モデルの重みとバイアスをGPUへ
model = model.to("cuda")


### 勾配計算について

# 勾配計算するテンソルにオプションを付ける
tensor = torch.tensor([[1, 2], [3, 4]], requires_grad=True)

# テンソル操作後に勾配を計算する
grad = tensor.backward()

# 変数を上書きすると requires_grad=True が外れるらしい
tensor = torch.tensor([[1, 2], [3, 4]], requires_grad=True)
tensor = tensor * 2
tensor.backward() → エラー
# 定義した変数が残っていれば、途中の操作 x などを上書きするのは問題ない

# 勾配のクリア
# 0ではなくNoneを入れる
# 細かい話だけど、新しく微分したときに += で値が入るが、Noneだと = で入る。後者の方が早い



