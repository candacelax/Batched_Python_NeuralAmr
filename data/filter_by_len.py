#quick script to filter by length a file:

MAX_LEN = 80
f = open("val.nl", "r")
f1 = open("val_short.nl", "w")
f2 = open("val_short.amr", "w")
f3 = open("val.amr", "r")

lines = zip(f.readlines(), f3.readlines())
for line in lines:
    if len(line[0].split(" ")) < MAX_LEN and len(line[1].split(" ")) < MAX_LEN:
        f1.write(line[0])
        f2.write(line[1])

f.close()
f1.close()
f2.close()
f3.close()
