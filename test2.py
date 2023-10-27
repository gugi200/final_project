

b = b''
test = [0, 1024, 1024, 1024, 1024]
for i, byte in enumerate(test):
    if i==0:
        b+= byte.to_bytes(1, 'little')
    else:
        b += byte.to_bytes(2, 'little')
print(list(b))
print(len(b))
print(b[1:3])
listByte = list(b)[1:]
print(listByte)

bv = b[1:]
sensorValues = [int.from_bytes(b[x:x+2], 'little') for x in range(1, 9, 2)]

print(sensorValues)


print([x for x in range(1, 9, 2)])
# stopByte1 = 255
# stopByte0 = 0
# expected = stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') + \
#             stopByte1.to_bytes(1, 'little') + stopByte0.to_bytes(1, 'little') 
# listTestByte = list(b)
# print(listTestByte)             # [2, 0, 48, 3, 53]
# listTestByteAsHex = [int(hex(x).split('x')[-1]) for x in listTestByte]
# print(listTestByteAsHex)        # [2, 0, 30, 3, 35]
