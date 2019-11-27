$("#recycle").hide();
$("#organic").hide();
$("#other").hide();

var video = $("#video");
var canvas = document.getElementById("canvas");
var context = canvas.getContext("2d");
var allowPhoto = false;

if (!navigator.mediaDevices.getUserMedia) {
    var p = navigator.mediaDevices.getUserMedia({
        video: true
    });
    p.then(function (mediaStream) {
        video.srcObject = mediaStream;
        video.onloadedmetadata = function (e) {
            // Do something with the video here.
            video.play();
        };
    });
    p.catch(function (err) { console.log(err.name); });
}

$("#button-classify").click(function () {
    $("#recycle").hide();
    $("#other").hide();
    $("#organic").hide();

    if (!allowPhoto) { // 标准的API
        var p = navigator.mediaDevices.getUserMedia({
            video: true
        });
        p.then(function (mediaStream) {
            var video = document.querySelector('video');
            allowPhoto = true;
            video.srcObject = mediaStream;
            video.onloadedmetadata = function (e) {
                // Do something with the video here.
                video.play();
            };
        });
        p.catch(function (err) { console.log(err.name); });
    } else {
        var canvasElement = document.getElementById("canvas");
        canvasElement.getContext("2d").drawImage(video.get(0), 0, 0, 320, 224);
        var MIME_TYPE = "image/jpg";
        var imgURL = canvasElement.toDataURL(MIME_TYPE);

        var dlLink = document.createElement('a');
        dlLink.download = "test1.jpg";
        dlLink.href = imgURL;
        dlLink.dataset.downloadurl = [MIME_TYPE, dlLink.download, dlLink.href].join(':');

        document.body.appendChild(dlLink);
        dlLink.click();
        document.body.removeChild(dlLink);
        setTimeout(function () {
            $.ajax({
                url: "http://127.0.0.1:5005/auto_classify",
                type: 'get',
                success: function (res) {
                    console.log("response=" + res);
                    if (res.result == "0") {
                        $("#other").show();
                        $("#recycle").hide();
                        $("#organic").hide();
                    } else if (res.result == "1") {
                        $("#organic").show();
                        $("#recycle").hide();
                        $("#other").hide();
                    } else {
                        $("#recycle").show();
                        $("#other").hide();
                        $("#organic").hide();
                    }
                },
            })
        }, 2000);
    }
})