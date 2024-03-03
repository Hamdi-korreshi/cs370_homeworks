from pytubefix import YouTube
import os


def grab_videos(links):
    # Grab the current directory
    curr_dir = os.getcwd()


    # Read video links from a file
    links = ['https://www.youtube.com/watch?v=wbWRWeVe1XE','https://www.youtube.com/watch?v=FlJoBhLnqko','https://www.youtube.com/watch?v=Y-bVwPRy_no']
    for link in links:
        # Create the YouTube object to access everything
        video = YouTube(link)

        # Create the captions file name and the folder name
        captions = video.title + " Captions"
        folder = "Videos"

        # Make the path for captions to be written to and video downloaded into
        folder_path = os.path.join(curr_dir, folder)

        if not os.path.exists(folder):
            os.mkdir(folder)

        
        video.streams.get_by_itag(22).download(output_path=folder_path)
        
        try:
            caption = video.captions['en']
        except:
            caption = video.captions['a.en']
        # Modify to make the file inside the folder
        write_captions = os.path.join(folder, captions + ".txt")
        file1 = open(write_captions, "w")

        # Make the file in a readable format
        file1.write(caption.generate_srt_captions())
        file1.close()
        print(caption)

        # Download video to the new folder

    print("DONE")
