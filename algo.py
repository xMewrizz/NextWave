flag = 'да'
while flag != 'нет':
    print('Введите элементы первого множества через пробел:')
    first_set = set(input().split())
    print('Введите элементы второго множества через пробел:')
    second_set = set(input().split())

    if first_set == second_set:
        print('Множества равны')
    if first_set != second_set:
        print('Множества не равны')
    print('Вы хотите продолжить?')
    flag = input().lower()
    