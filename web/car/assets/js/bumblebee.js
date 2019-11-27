$("#button-car-waypoint").click(function(){
    console.log("DRIVE CLICKED");
    $.ajax({
        url: "http://10.42.0.1:8081/gotowaypoint?name=A",
        type: 'get',
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        success: function(res){
            console.log("DRIVE SUCCESS");
        },
        timeout:3000
    })                                
})

$("#button-car-photo").click(function(){
    console.log("PHOTO CLICKED");
    $.ajax({
        url: "http://10.42.0.1:5000/image",
        type: 'get',
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        success: function(res){
            console.log("PHOTO SUCCESS")  
        },
        timeout:3000
    })                                
})

