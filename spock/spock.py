"""Main module."""
socket_mode_client.connect()
while True:
    
    with open('json/ouput.json', 'r') as file:
        scholars_publications = json.load(file)

    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:  # Adjust max_workers as needed
        executor.map(process_scholar, scholars_publications.items())
    

    print('Waiting!')
    time.sleep(900)
