import torch
from timeit import timeit
def test():
	torch.manual_seed(2)
	a = torch.rand((200,200))
	b = torch.rand((200,200))
	c = torch.matmul(a, b)
	print(float(c[0,0].numpy()))
test()
#print(timeit(test, number=1))
