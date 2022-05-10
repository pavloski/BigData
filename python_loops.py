roster = []
while True:
    name = input("the name: ")
    if len(name) < 1:
        break
    elif name[0].lower() in "aeiuo":
        roster.append(name)

print(roster)