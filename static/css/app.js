$(document).ready(function() {
    setInterval(ajaxd, 10000);
});

function ajaxd() { 
  $.getJSON('/getDisplayData', function(data) {
    console.log(data)
      $("#success").val(data.boxes); //msg wohi jo console.log kr rahay thay
      // $("#success").load(" #success > *");
      // var tag = document.getElementById("success");
      // tag.innerHTML = data;
  });
   }
     