  async function takePhoto() {
        const capture = document.getElementById('captureButton');
        $("#camerStartButton").attr("disabled", "true")
        $("#video").show();
        $("#captureButton").show();
        // $("#uploadButton").hide();
        // $("#uploadORLabel").hide();
        $("#captureButton").show();

        const video = document.getElementById('video');
        try{
          const stream = await navigator.mediaDevices.getUserMedia({video: true});

          video.srcObject = stream;
          await video.play();

          await new Promise((resolve) => capture.onclick = resolve);
          const canvas = document.getElementById('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        $("#cavas").hide();
        $("#camerStartButton").hide();
        $("#video").hide();
        $(".loadingText").show();

        return {'w': canvas.width, 'h': canvas.height, img: canvas.toDataURL('image/jpeg')};
        }catch(e){
          $("#cameraPermissionText").show();
          $(".loadingText").hide();              
            $("#retryButton").show();
            $("#captureButton").hide();
            return;
        }

        
      }

      function setResultImage(encoding){
        // encoding = window.btoa(encoding)

        // data = 'data:image/jpeg;base64,' + encoding
        // console.log(data);

        document.getElementById("resImg").src = encoding;
      }

      function extractRect(rect){
        let res = []
        rect = rect.substring(1, rect.length - 1)
        res = rect.split(' ')
        return res
      }

      async function saveIGMG(){
        const photoData = await takePhoto();
        if (!photoData){
          return
        }
        const img = photoData.img;
        const reqBody = {image: img}

        try{
          // fetch("http://127.0.0.1:5000/upload",{
         fetch("https://t3p7tb2ifc.execute-api.us-east-2.amazonaws.com/FinalStage", {
            method: "POST",
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(reqBody)
          }).then(
            (preJson) => {
              try{
              preJson.json()
            .then((data)=>{
              // console.log(data.rect);

              const rect = extractRect(data.rect);
              // console.log(rect);
              const canvas = document.getElementById("canvas")
              const ctx = canvas.getContext("2d");

              var image = new Image();
              image.onload = function() {
                canvas.height = photoData.h;
                canvas.width = photoData.w;
                ctx.drawImage(image, 0, 0);
                ctx.beginPath();
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 5;
                ctx.roundRect(parseInt(rect[0]), parseInt(rect[1]), parseInt(rect[2]), parseInt(rect[3]), 5);
                ctx.stroke();
                document.getElementById("emotionLabel").innerText = "Emotion:";
                document.getElementById("emotionValue").innerText = data.emotion;
                $("#captureButton").hide();
                $("#retryButton").show();
                $("#canvas").show();
                $("#emotionLabel").show();
                $("#emotionValue").show();
                $("#camerStartButton").attr("disabled", false);
                $(".loadingText").hide();              

              };
              image.src = img
              
            } )
            .catch((er)=>{
              $(".loadingText").hide();              
              $("#faceError").show();
              $("#retryButton").show();
              $("#captureButton").hide();
            })
          }catch(e){
            $(".loadingText").hide();              
            $("#faceError").show();
            $("#retryButton").show();
            $("#captureButton").hide();          }
          }
          ).catch((serr)=>{
            $(".loadingText").hide();              
            $("#serverError").show();
            $("#retryButton").show();
            $("#captureButton").hide();
          })
        }catch(err){
          // alert("wat")
            $(".loadingText").hide();              
            $("#serverError").show();
            $("#retryButton").show();
            $("#captureButton").hide();
        }

      }

function resetCameraScreen(){
  $("#camerStartButton").show();
  $("#camerStartButton").attr("disabled", false);
  $("#captureButton").hide();
  $("#video").hide();
  $("#retryButton").hide();
  $("#resCanvas").hide(); 
  $("#canvas").hide(); 
  $("#emotionLabel").hide(); 
  $("#emotionValue").hide(); 
  $(".loadingText").hide();
  $(".serverError").hide();  
  $("#serverError").hide();  
  $("#cameraPermissionText").hide();          
  $("#faceError").hide();              


  // $("#uploadButton").show();
  // $("#uploadORLabel").show();
}


$(document).ready(()=>{
  // window.scrollTo(0,1);
  const vh = window.innerHeight * 0.01;
  document.documentElement.style.setProperty('--vh', `${vh}px`);

  $("#startButton").click(()=>{
    $("#page1").fadeOut("fast", complete=
      ()=>{
        $("#page2").css('display', 'flex')
        $("#page2").fadeIn("fast")
      }
    );
  })


  $("#retryButton").click(()=>{
    resetCameraScreen();
  })
  

  $("#backButton2").click(()=>{
    $("#page2").fadeOut("fast", complete=
      ()=>{
        $("#page1").css('display', 'flex')
        $("#page1").fadeIn("fast")
      }
    );
  })

  $("#backButton3").click(()=>{
    $("#page3").fadeOut("fast", complete=
      ()=>{
        $("#page2").css('display', 'flex')
        $("#page2").fadeIn("fast")
      }
    );
  })

  $("#nextButton2").click(()=>{
    $("#page2").fadeOut("fast", complete=
      ()=>{
        $("#page3").css('display', 'flex')
        $("#page3").fadeIn("fast")
      }
    );
  })

  $("#hideLearnMoreButton").click(()=>{
    $("#hideLearnMoreButton").hide();
    $("#learnMore").slideUp();
  })

  $("#learnMoreButton").click(()=>{
    $("#hideLearnMoreButton").show();
    $("#learnMore").slideDown();
  })
  
})