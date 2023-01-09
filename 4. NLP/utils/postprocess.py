def concat_tag(iob_format):
    word_list = [item[0] for item in iob_format]
    tag_list = [item[1] for item in iob_format]

    new_word_list = []
    new_tag_list = []

    for i in range(len(word_list)):
        word = word_list[i]
        tag = tag_list[i]

        if tag == 'O':
            new_word_list.append(word)
            new_tag_list.append(tag)
        else:
            tag_loc = tag.split('-')[0]
            tag = tag.split('-')[1]

            if tag_loc == 'B':
                new_word_list.append(word)
                new_tag_list.append(tag)
            elif tag_loc == 'I':
                pre_word = new_word_list[-1]
                new_word_list[-1] = (pre_word + ' ' + word)

    return new_word_list, new_tag_list
