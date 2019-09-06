files = ["train", "val", "test"]
for name in files:
    new_amr = open("new_" + name + ".amr", "w")
    new_nl = open("new_" + name + ".nl", "w")
    new_align = open("new_" + name + ".align", "w")

    old_amr = open(name + ".amr", "r")
    old_nl = open(name + ".nl", "r")
    old_align = open(name + ".align", "r")
    
    old_amr_lines = old_amr.readlines()
    old_nl_lines = old_nl.readlines()
    old_align_lines = old_align.readlines()
    for amr, nl, align in zip(old_amr_lines, old_nl_lines, old_align_lines):
        if (not amr.startswith("multi-sentence")) and len(nl.split(" "))<100:
            new_amr.write(amr.replace("/",""))
            new_nl.write(nl) 
            new_align.write(align) 
    new_amr.close()
    new_nl.close()
    old_amr.close()
    old_nl.close()
