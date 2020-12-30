$(document).ready(function() {
  setInterval(ajaxd, 1000);
  $(".loader").hide();
  $("#hideload").click(function(){
    $(".loader").show();
    $("#hideload").hide();
  });
});

function ajaxd() { 
$.getJSON('/getDisplayData', function(data) {
  console.log(data)
    $("#success").text(data.boxes); //msg wohi jo console.log kr rahay thay
    // $("#success").load(" #success > *");
    // var tag = document.getElementById("success");
    // tag.innerHTML = data;
});
 }

