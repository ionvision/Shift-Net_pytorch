
import torch
import torch.nn as nn

import functools


'''
Code insipired from the paper 'Globally and Locally Consistent Image Completion'.
http://hi.cs.waseda.ac.jp/~iizuka/projects/completion/
'''
class CompletionGenerator(nn.Module):

	def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_dropout=False,
							 gpu_ids=[], ngf=64):
		super (CompletionGenerator, self).__init__ ()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.gpu_ids = gpu_ids

		print('[INFO] INPUT_NC {}'.format(input_nc))
		print('[INFO] OUTPUT_NC {}'.format (output_nc))
		self.norm_layer = norm_layer
		self.use_dropout = use_dropout
		self.gpu_ids = gpu_ids

		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		# Input (5x5 , dilation = 1, stride = 1)
		model = [nn.ReflectionPad2d(2),
		         nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0,
						   bias=use_bias),
				 norm_layer(ngf),
				 nn.ReLU(True)]

		# Input (3x3 , dilation = 1, stride = 2 and then 1)
		stride = 2
		for i in range(1, 3):
			if i % 2 == 0:
				stride = 1
			model += [nn.Conv2d (i*ngf, 2*i*ngf, kernel_size=3, padding=(1, 1),
			                    bias=use_bias, stride=stride),
			         norm_layer (ngf),
			         nn.ReLU (True)]

		# Input (3x3 , dilation = 1, stride = 2)
		ngf *= 4
		model += [nn.Conv2d (ngf, ngf, kernel_size=3, padding=(1, 1),
		                     bias=use_bias, stride=2),
		          norm_layer (ngf),
		          nn.ReLU (True)]

		dilation = 2
		for i in range(5):
			print(dilation**i)
			model += [nn.Conv2d (ngf, ngf, kernel_size=3, padding=(dilation**i, dilation**i),
			                     bias=use_bias, dilation=dilation**i),
			          norm_layer (ngf),
			          nn.ReLU (True)]

		for _ in range(2):
			model += [nn.Conv2d (ngf, ngf, kernel_size=3, padding=(1, 1),
			                     bias=use_bias),
			          norm_layer (ngf),
			          nn.ReLU (True)]
		ngf = ngf//2

		model += [nn.ConvTranspose2d (2* ngf, ngf,
		                              kernel_size=4, stride=2,
		                              padding=1, output_padding=0,
		                              bias=use_bias),
		          norm_layer (ngf),
		          nn.ReLU (True)]

		model += [nn.Conv2d (ngf, ngf, kernel_size=3, padding=(1, 1),
		                     bias=use_bias),
		          norm_layer (ngf),
		          nn.ReLU (True)]

		ngf = ngf//2

		model += [nn.ConvTranspose2d (2* ngf, ngf,
		                              kernel_size=4, stride=2,
		                              padding=1, output_padding=0,
		                              bias=use_bias),
		          norm_layer (ngf),
		          nn.ReLU (True)]

		model += [nn.Conv2d (ngf, ngf, kernel_size=3, padding=1,
		                     bias=use_bias),
		          norm_layer (ngf),
		          nn.ReLU (True)]

		model += [nn.Conv2d(ngf, output_nc, kernel_size=3, padding=1)]
		model += [nn.Tanh()]

		self.model = nn.Sequential (*model)

	def forward(self, input):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		else:
			return self.model(input)
