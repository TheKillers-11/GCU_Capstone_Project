# GCU_Capstone_Project
Benjamin Rudin's Computer Science Masters Capstone Application

Please see the "requirements.txt" to install the necessary Python version / libraries needed to run the code. 

The following section is included at the end of the "BenjaminRudinCapstoneApplicationCode.py" file in this repository:

# Main function to run the Dash application / server
if __name__ == "__main__":
    #app.run_server(debug=True) # THIS LINE OR SOMETHING SIMILAR WILL BE USED IN OTHER IDE's; NOTE THAT THE APP IS FORMATTED FOR EXTERNAL WINDOWS
    # I HAVE ONLY USED JUPYTER_LABS FOR THIS ASSIGNMENT; I CANNOT SPEAK ON OTHER IDE's
    #app.run_server(jupyter_mode='external',port=7953,debug=True)
    
    # To run this application on Jupyter without the Dash debug pop-up, remove debug=True
    app.run_server(jupyter_mode='external',port=7953)

Please note, the "app.run_server(jupyter_mode='external',port=7953)" line must be used if running the code in JupyterLab. If not, comment out this line and uncomment the "#app.run_server(debug=True)" for non-Jupyter .py environments. 

Thank you!
