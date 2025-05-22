##     PoliticalBiasChecker: --> know before you read


def PoliticalBiasChecker(bias_rate):

    belief = input("Optional [What are your political beliefs (answer right, left, other)?]")
    
    while belief.lower() not in ('right', 'left', 'other'):
        belief = input("Try again following all naming standards: [What are your political beliefs (answer right, left, other)?]")
    
    if belief == 'right':
        print("Significant biased words: ", important_words)
        print("Extremely significant biased words: ", extremely_important_words)
        
        if dom_bias == "Right-biased":
            if bias_rate >= 0.3:
                print("Article is overly biased for your beliefs.")
            else:
                print("Article is not too biased for you.")
            
    elif belief == 'left':
        print("Significant biased words: ", important_words)
        print("Extremely significant biased words: ", extremely_important_words)
    
        if dom_bias == "Left-biased":
            if bias_rate >= 0.3:
                print("Article is overly biased for your beliefs.")
            else:
                print("Article is not too biased for you.")
                
            
    else:
        print("Significant biased words: ", important_words)
        print("Extremely significant biased words: ", extremely_important_words)
        if dom_bias not in ('Left-biased', 'Right-biased'): #either unclear/balanced or none
            if bias_rate >= 0.3:
                print("Article is overly biased for your beliefs.")
            else:
                print("Article is not too biased for you.")
            
        

choice = input("Would you like a personalized bias detection, according to your beliefs? Enter Y or N")

while choice.lower() not in ('y', 'n'):
    choice = input("Wrong input. Only enter Y or N")

if choice.upper() == 'Y':
    personal_stats = PoliticalBiasChecker(bias_rate)
    print(personal_stats)
else:
    print("Ok. You may continue with other functionalities of the PoliticalBias model.")
