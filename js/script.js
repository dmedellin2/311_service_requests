$("#complaint_type").change ( function () {
    var targID  = $(this).val ();
    $("div.style-sub-1").hide ();
    $('#' + targID).show ();
} )
