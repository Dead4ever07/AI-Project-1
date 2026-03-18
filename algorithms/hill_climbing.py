from cost import cost_function


def run(submission, family_size, choices):

    best = submission["assigned_day"].values
    best_score = cost_function(best, family_size, choices)

    new = best.copy()

    for fam_id in range(len(best)):
        for pick in range(10):

            day = choices[fam_id, pick]

            temp = new.copy()
            temp[fam_id] = day

            score = cost_function(temp, family_size, choices)

            if score < best_score:
                new = temp.copy()
                best_score = score

    return new, best_score
