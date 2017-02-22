require 'torch'

local a = torch.load('features.t7')
# print (#a)
# print (table.getn(a))
local count = 0
for k,v in pairs(a) do
	count = count + 1
end
# print (count)

data = torch.DoubleTensor(count, 2048):zero()
i = 1
imgpath = {}
for k,v in pairs(a) do
	data[{i,{} }]:copy(v)
	i = i + 1
	print (k)
	table.insert(imgpath, k)
	-- print (v)
end
local matio = require 'matio'

matio.save('features.mat', data)

